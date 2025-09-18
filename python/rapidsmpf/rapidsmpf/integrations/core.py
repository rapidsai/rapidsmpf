# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffler integration with external libraries."""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass, field
from functools import partial
from numbers import Number  # noqa: TC003
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, TypeVar

import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmpf.buffer.buffer import MemoryType
from rapidsmpf.buffer.resource import BufferResource, LimitAvailableMemory
from rapidsmpf.buffer.spill_collection import SpillCollection
from rapidsmpf.config import (
    Optional,
    OptionalBytes,
    Options,
)
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.shuffler import Shuffler
from rapidsmpf.statistics import Statistics

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rapidsmpf.communicator.communicator import Communicator


DataFrameT = TypeVar("DataFrameT")


# Set of available shuffle IDs
_shuffle_id_vacancy: set[int] = set(range(Shuffler.max_concurrent_shuffles))
_shuffle_id_vacancy_lock: threading.Lock = threading.Lock()


def get_new_shuffle_id(get_occupied_ids: Callable[[], Sequence[set[int]]]) -> int:
    """
    Get a new available shuffle ID.

    Since RapidsMPF only supports a limited number of shuffler instances at
    any given time, this function maintains a shared pool of shuffle IDs.

    If no IDs are available locally, it calls get_occupied_ids to query all
    workers for IDs in use, updates the vacancy set accordingly, and retries.
    If all IDs are in use across all workers, an error is raised.

    Parameters
    ----------
    get_occupied_ids
        Callable function that returns the occupied shuffle IDs.

    Returns
    -------
    A unique shuffle ID not currently in use.

    Raises
    ------
    ValueError
        If all shuffle IDs are currently in use.
    """
    global _shuffle_id_vacancy  # noqa: PLW0603

    with _shuffle_id_vacancy_lock:
        if not _shuffle_id_vacancy:
            # We start with setting all IDs as vacant and then subtract all
            # IDs occupied on any one worker.
            _shuffle_id_vacancy = set(range(Shuffler.max_concurrent_shuffles))
            _shuffle_id_vacancy.difference_update(*get_occupied_ids())
            if not _shuffle_id_vacancy:
                raise ValueError(
                    f"Cannot shuffle more than {Shuffler.max_concurrent_shuffles} "
                    "times in a single query."
                )

        return _shuffle_id_vacancy.pop()


@dataclass
class WorkerContext:
    """
    RapidsMPF specific attributes for a worker.

    Attributes
    ----------
    lock
        The global worker lock. Must be acquired before accessing attributes
        that might be modified while the worker is running such as the shufflers.
    br
        The buffer resource used by the worker exclusively.
    progress_thread
        The progress thread used by the worker.
    comm
        The communicator connected to all other workers.
    statistics
        The statistics used by the worker. If None, statistics is disabled.
    spill_collection
        A collection of Python objects that can be spilled to free up device memory.
    shufflers
        A mapping from shuffler IDs to active shuffler instances.
    options
        Configuration options.
    """

    lock: ClassVar[threading.RLock] = threading.RLock()
    br: BufferResource
    progress_thread: ProgressThread
    comm: Communicator
    statistics: Statistics
    spill_collection: SpillCollection = field(default_factory=SpillCollection)
    shufflers: dict[int, Shuffler] = field(default_factory=dict)
    options: Options = field(default_factory=Options)

    def get_statistics(self) -> dict[str, dict[str, Number]]:
        """
        Get the statistics from the worker context.

        Returns
        -------
        statistics
            A dictionary of statistics. The keys are the names of the statistics.
            The values are dictionaries with two keys:

            - "count" is the number of times the statistic was recorded.
            - "value" is the value of the statistic.

        Notes
        -----
        Statistics are global across all shuffles. To measure statistics for any
        given shuffle, gather statistics before and after the shuffle and compute
        the difference.
        """
        return {
            stat: self.statistics.get_stat(stat)
            for stat in self.statistics.list_stat_names()
        }


class ShufflerIntegration(Protocol[DataFrameT]):
    """Shuffle-integration protocol."""

    @staticmethod
    def insert_partition(
        df: DataFrameT,
        partition_id: int,
        partition_count: int,
        shuffler: Shuffler,
        options: Any,
        *other: Any,
    ) -> None:
        """
        Add a partition to a RapidsMPF Shuffler.

        Parameters
        ----------
        df
            DataFrame partition to add to a RapidsMPF shuffler.
        partition_id
            The input partition id of ``df``.
        partition_count
            Number of output partitions for the current shuffle.
        shuffler
            The RapidsMPF Shuffler object to extract from.
        options
            Additional options.
        *other
            Other data needed for partitioning. For example,
            this may be boundary values needed for sorting.
        """
        ...

    @staticmethod
    def extract_partition(
        partition_id: int,
        shuffler: Shuffler,
        options: Any,
    ) -> DataFrameT:
        """
        Extract a DataFrame partition from a RapidsMPF Shuffler.

        Parameters
        ----------
        partition_id
            Partition id to extract.
        shuffler
            The RapidsMPF Shuffler object to extract from.
        options
            Additional options.

        Returns
        -------
        A shuffled DataFrame partition.
        """
        ...


def get_shuffler(
    ctx: WorkerContext,
    shuffle_id: int,
    *,
    partition_count: int | None = None,
    worker: Any = None,
) -> Shuffler:
    """
    Return the appropriate :class:`Shuffler` object.

    Parameters
    ----------
    ctx
        The worker context.
    shuffle_id
        Unique ID for the shuffle operation.
    partition_count
        Output partition count for the shuffle operation.
    worker
        The current worker.

    Returns
    -------
    The active RapidsMPF :class:`Shuffler` object associated with
    the specified ``shuffle_id``, ``partition_count`` and
    ``worker``.

    Notes
    -----
    Whenever a new :class:`Shuffler` object is created, it is
    saved as ``WorkerContext.shufflers[shuffle_id]``.
    """
    with ctx.lock:
        if shuffle_id not in ctx.shufflers:
            if partition_count is None:
                raise ValueError(
                    "Need partition_count to create new shuffler."
                    f" shuffle_id: {shuffle_id}\n"
                    f" Shufflers: {ctx.shufflers}"
                )
            assert ctx.br is not None
            assert ctx.comm is not None
            assert ctx.progress_thread is not None
            ctx.shufflers[shuffle_id] = Shuffler(
                ctx.comm,
                ctx.progress_thread,
                op_id=shuffle_id,
                total_num_partitions=partition_count,
                stream=DEFAULT_STREAM,
                br=ctx.br,
                statistics=ctx.statistics,
            )
        return ctx.shufflers[shuffle_id]


def insert_partition(
    get_context: Callable[..., WorkerContext],
    callback: Callable[
        [
            DataFrameT,
            int,
            int,
            Shuffler,
            Any,
            *tuple[str | tuple[str, int], ...],
        ],
        None,
    ],
    df: DataFrameT,
    partition_id: int,
    partition_count: int,
    shuffle_id: int,
    options: Any,
    *other_keys: str | tuple[str, int],
) -> None:
    """
    Add a partition to a RapidsMPF Shuffler.

    Parameters
    ----------
    get_context
        Callable function to fetch the worker context.
    callback
        Insertion callback function. This function must be the
        `insert_partition` attribute of a `ShufflerIntegration`
        protocol.
    df
        DataFrame partition to add to a RapidsMPF shuffler.
    partition_id
        The input partition id of ``df``.
    partition_count
        Number of output partitions for the current shuffle.
    shuffle_id
        The RapidsMPF shuffle id.
    options
        Optional key-word arguments.
    *other_keys
        Other keys needed by ``callback``.
    """
    callback(
        df,
        partition_id,
        partition_count,
        get_shuffler(get_context(), shuffle_id),
        options,
        *other_keys,
    )


def extract_partition(
    get_context: Callable[..., WorkerContext],
    callback: Callable[
        [int, Shuffler, Any],
        DataFrameT,
    ],
    shuffle_id: int,
    partition_id: int,
    worker_barrier: tuple[int, ...],
    options: Any,
) -> DataFrameT:
    """
    Extract a partition from a RapidsMPF Shuffler.

    Parameters
    ----------
    get_context
        Callable function to fetch the worker context.
    callback
        Insertion callback function. This function must be the
        `extract_partition` attribute of a `ShufflerIntegration`
        protocol.
    shuffle_id
        The RapidsMPF shuffle id.
    partition_id
        Partition id to extract.
    worker_barrier
        Worker-barrier task dependency. This value should
        not be used for compute logic.
    options
        Additional options.

    Returns
    -------
    Extracted DataFrame partition.
    """
    shuffler = get_shuffler(get_context(), shuffle_id)
    try:
        return callback(
            partition_id,
            shuffler,
            options,
        )
    finally:
        if shuffler.finished():
            ctx = get_context()
            with ctx.lock:
                if shuffle_id in ctx.shufflers:
                    del ctx.shufflers[shuffle_id]


class JoinIntegration(Protocol[DataFrameT]):
    """Join-integration protocol."""

    @staticmethod
    def shuffler_integration() -> ShufflerIntegration[DataFrameT]:
        """Return the shuffler integration."""
        ...

    @classmethod
    def join_partition(
        cls,
        ctx: WorkerContext,
        bcast_side: Literal["left", "right", "none"],
        left: int | DataFrameT,
        right: int | DataFrameT,
        part_id: int,
        n_worker_tasks: int,
        options: Any,
    ) -> DataFrameT:
        """
        Produce a joined table partition.

        Parameters
        ----------
        ctx
            The worker context.
        bcast_side
            The side of the join being broadcasted. If "none", this is
            a regular hash join.
        left
            The left-table operation id or the left partition.
            The operation may correspond to an allgather or shuffle operation.
        right
            The right-table operation id or the right partition.
            The operation may correspond to an allgather or shuffle operation.
        part_id
            The output partition id.
        n_worker_tasks
            The number of join_partition tasks to be called on this worker.
            This information may be used for cleanup.
        options
            Additional options.

        Returns
        -------
        A joined DataFrame chunk.

        Notes
        -----
        This method is used to produce a single joined table chunk.
        """
        ...


def join_partition(
    get_context: Callable[..., WorkerContext],
    callback: Callable[
        [
            WorkerContext,  # ctx
            Literal["left", "right", "none"],  # bcast_side
            int | DataFrameT,  # left
            int | DataFrameT,  # right
            int,  # part_id
            int,  # n_worker_tasks
            Any,  # options
        ],
        DataFrameT,
    ],
    bcast_side: Literal["left", "right", "none"],
    left_op_id: int | None,
    right_op_id: int | None,
    left_barrier: DataFrameT | tuple[int, ...],
    right_barrier: DataFrameT | tuple[int, ...],
    part_id: int,
    n_worker_tasks: int,
    options: Any,
) -> DataFrameT:
    """
    Produce a joined table partition.

    Parameters
    ----------
    get_context
        Callable function to fetch the worker context.
    callback
        Join callback function. This function must be the
        `join_partition` attribute of a `JoinIntegration`
        protocol.
    bcast_side
        The side of the join being broadcasted. If "none", this is
        a regular hash join.
    left_op_id
        The left-table operation id. The operation may correspond
        to an allgather or a shuffle operation. If None, the
        left_barrier argument must be the left partition.
    right_op_id
        The right-table operation id. The operation may correspond
        to an allgather or a shuffle operation. If None, the
        right_barrier argument must be the right partition.
    left_barrier
        Worker-barrier task dependency for the left table or the left partition.
    right_barrier
        Worker-barrier task dependency for the right table or the right partition.
    part_id
        The output partition id.
        This information is needed to extract shuffled partitions.
    n_worker_tasks
        The number of join_partition tasks to be called on this worker.
        This information may be used for cleanup.
    options
        Additional options.
    """
    ctx = get_context()
    left: int | DataFrameT
    if left_op_id is None:
        assert not isinstance(left_barrier, tuple)
        left = left_barrier
    else:
        left = left_op_id
    right: int | DataFrameT
    if right_op_id is None:
        assert not isinstance(right_barrier, tuple)
        right = right_barrier
    else:
        right = right_op_id
    return callback(
        ctx,
        bcast_side,
        left,
        right,
        part_id,
        n_worker_tasks,
        options,
    )


# Create a spill function that spills the python objects in the spill-
# collection. This way, we have a central place (the worker) to track
# and trigger spilling of python objects.
def spill_func(
    amount: int,
    *,
    staging_buffer: rmm.DeviceBuffer | None,
    lock: threading.Lock,
    mr: rmm.mr.DeviceMemoryResource,
    ctx: WorkerContext,
) -> int:
    """
    Spill a specified amount of data from the Python object spill collection.

    This function attempts to use a preallocated staging device buffer to
    spill Python objects from the spill collection. If the staging buffer
    is currently in use, it will fall back to spilling without it.

    Parameters
    ----------
    amount
        The amount of data to spill, in bytes.
    staging_buffer
        Optional buffer to stage data through.
    lock
        Lock to protect access.
    mr
        Memory resource for device allocations.
    ctx
        The worker context to spill from.

    Returns
    -------
    The actual amount of data spilled, in bytes.
    """
    if staging_buffer is not None and lock.acquire(blocking=False):
        try:
            return ctx.spill_collection.spill(
                amount,
                stream=DEFAULT_STREAM,
                device_mr=mr,
                staging_device_buffer=staging_buffer,
            )
        finally:
            lock.release()
    return ctx.spill_collection.spill(amount, stream=DEFAULT_STREAM, device_mr=mr)


def rmpf_worker_setup(
    worker: Any,
    option_prefix: str,
    *,
    comm: Communicator,
    options: Options,
) -> WorkerContext:
    """
    Attach RapidsMPF shuffling attributes to a worker process.

    Parameters
    ----------
    worker
        The current worker process.
    option_prefix
        Prefix for config-option names.
    comm
        Communicator for shufflers.
    options
        Configuration options.

    Returns
    -------
    WorkerContext
        New worker context set up for shuffling.

    Warnings
    --------
    This function creates a new RMM memory pool, and
    sets it as the current device resource.
    """
    # Insert RMM resource adaptor on top of the current RMM resource stack.
    mr = RmmResourceAdaptor(
        upstream_mr=rmm.mr.get_current_device_resource(),
        fallback_mr=(
            # Use a managed memory resource if OOM protection is enabled.
            rmm.mr.ManagedMemoryResource()
            if options.get_or_default(
                f"{option_prefix}oom_protection", default_value=False
            )
            else None
        ),
    )
    rmm.mr.set_current_device_resource(mr)

    # Print statistics at worker shutdown.
    if options.get_or_default(f"{option_prefix}statistics", default_value=False):
        statistics = Statistics(enable=True, mr=mr)
    else:
        statistics = Statistics(enable=False)

    if (
        options.get_or_default(f"{option_prefix}print_statistics", default_value=True)
        and statistics.enabled
    ):
        weakref.finalize(
            worker,
            lambda name, stats: print(name, stats.report()),
            name=str(worker),
            stats=statistics,
        )

    # Create a buffer resource with a limiting availability function.
    total_memory = rmm.mr.available_device_memory()[1]
    spill_device = options.get_or_default(
        f"{option_prefix}spill_device", default_value=0.50
    )
    memory_available = {
        MemoryType.DEVICE: LimitAvailableMemory(
            mr, limit=int(total_memory * spill_device)
        )
    }
    br = BufferResource(
        mr,
        memory_available=memory_available,
        periodic_spill_check=options.get_or_default(
            f"{option_prefix}periodic_spill_check", default_value=Optional(1e-3)
        ).value,
    )

    # If enabled, create a staging device buffer for the spilling to reduce
    # device memory pressure.
    # TODO: maybe have a pool of staging buffers?
    spill_staging_buffer_size = options.get_or_default(
        f"{option_prefix}staging_spill_buffer",
        default_value=OptionalBytes("128 MiB"),
    ).value
    spill_staging_buffer = (
        None
        if spill_staging_buffer_size is None
        else rmm.DeviceBuffer(
            size=spill_staging_buffer_size, stream=DEFAULT_STREAM, mr=mr
        )
    )
    ctx = WorkerContext(
        br=br,
        progress_thread=ProgressThread(comm, statistics),
        comm=comm,
        statistics=statistics,
        options=options,
    )

    # Add the spill function using a negative priority (-10) such that spilling
    # of internal shuffle buffers (non-python objects) have higher priority than
    # spilling of the Python objects in the collection.
    br.spill_manager.add_spill_function(
        func=partial(
            spill_func,
            staging_buffer=spill_staging_buffer,
            lock=threading.Lock(),
            mr=mr,
            ctx=ctx,
        ),
        priority=-10,
    )
    return ctx
