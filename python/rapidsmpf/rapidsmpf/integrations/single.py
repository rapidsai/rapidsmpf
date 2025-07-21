# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffler integration for single-worker pylibcudf execution."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, cast

import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmpf.buffer.buffer import MemoryType
from rapidsmpf.buffer.resource import BufferResource, LimitAvailableMemory
from rapidsmpf.communicator.single import new_communicator
from rapidsmpf.config import Optional, OptionalBytes, Options
from rapidsmpf.integrations.core import ShufflerIntegration, WorkerContext
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.shuffler import Shuffler

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rapidsmpf.integrations.core import DataFrameT


# Set of available shuffle IDs
_shuffle_id_vacancy: set[int] = set(range(Shuffler.max_concurrent_shuffles))
_shuffle_id_vacancy_lock: threading.Lock = threading.Lock()


# Local single-worker context
class _SingleWorker:
    """Mutable single-worker utility class."""

    context: WorkerContext | None = None


_single_rapidsmpf_worker: _SingleWorker = _SingleWorker()


def get_single_worker_context() -> WorkerContext:
    """
    Retrieve the single `WorkerContext`.

    If the worker context does not already exist on the worker, it
    will be created and attached to `_single_rapidsmpf_worker_context`.

    Returns
    -------
    The existing or newly initialized worker context.
    """
    with WorkerContext.lock:
        if _single_rapidsmpf_worker.context is None:
            _single_rapidsmpf_worker.context = WorkerContext()
        return cast(WorkerContext, _single_rapidsmpf_worker.context)


def setup_single_worker(options: Options = Options()) -> None:
    """
    Attach RapidsMPF shuffling attributes to a single worker.

    Parameters
    ----------
    options
        Configuration options.

    Warnings
    --------
    This function creates a new RMM memory pool, and
    sets it as the current device resource.
    """
    ctx = get_single_worker_context()
    with ctx.lock:
        if ctx.comm is not None:
            return  # Single worker already set up

        ctx.options = options
        ctx.comm = new_communicator(options)
        ctx.comm.logger.trace("single communicator created.")

        # Insert RMM resource adaptor on top of the current RMM resource stack.
        mr = RmmResourceAdaptor(
            upstream_mr=rmm.mr.get_current_device_resource(),
            fallback_mr=(
                # Use a managed memory resource if OOM protection is enabled.
                rmm.mr.ManagedMemoryResource()
                if ctx.options.get_or_default(
                    "dask_oom_protection", default_value=False
                )
                else None
            ),
        )
        rmm.mr.set_current_device_resource(mr)

        ctx.progress_thread = ProgressThread(ctx.comm, ctx.statistics)

        # Create a buffer resource with a limiting availability function.
        total_memory = rmm.mr.available_device_memory()[1]
        spill_device = ctx.options.get_or_default(
            "single_spill_device", default_value=0.50
        )
        memory_available = {
            MemoryType.DEVICE: LimitAvailableMemory(
                mr, limit=int(total_memory * spill_device)
            )
        }
        ctx.br = BufferResource(
            mr,
            memory_available=memory_available,
            periodic_spill_check=ctx.options.get_or_default(
                "single_periodic_spill_check", default_value=Optional(1e-3)
            ).value,
        )

        # If enabled, create a staging device buffer for the spilling to reduce
        # device memory pressure.
        # TODO: maybe have a pool of staging buffers?
        spill_staging_buffer_size = ctx.options.get_or_default(
            "dask_staging_spill_buffer",
            default_value=OptionalBytes("128 MiB"),
        ).value
        spill_staging_buffer = (
            None
            if spill_staging_buffer_size is None
            else rmm.DeviceBuffer(
                size=spill_staging_buffer_size, stream=DEFAULT_STREAM, mr=mr
            )
        )
        spill_staging_buffer_lock = threading.Lock()

        # Create a spill function that spills the python objects in the spill-
        # collection. This way, we have a central place (the dask worker) to track
        # and trigger spilling of python objects.
        def spill_func(amount: int) -> int:
            """
            Spill a specified amount of data from the Python object spill collection.

            This function attempts to use a preallocated staging device buffer to
            spill Python objects from the spill collection. If the staging buffer
            is currently in use, it will fall back to spilling without it.

            Parameters
            ----------
            amount
                The amount of data to spill, in bytes.

            Returns
            -------
            The actual amount of data spilled, in bytes.
            """
            if spill_staging_buffer is not None and spill_staging_buffer_lock.acquire(
                blocking=False
            ):
                try:
                    return ctx.spill_collection.spill(
                        amount,
                        stream=DEFAULT_STREAM,
                        device_mr=mr,
                        staging_device_buffer=spill_staging_buffer,
                    )
                finally:
                    spill_staging_buffer_lock.release()
            return ctx.spill_collection.spill(
                amount, stream=DEFAULT_STREAM, device_mr=mr
            )

        # Add the spill function using a negative priority (-10) such that spilling
        # of internal shuffle buffers (non-python objects) have higher priority than
        # spilling of the Python objects in the collection.
        ctx.br.spill_manager.add_spill_function(func=spill_func, priority=-10)


def _get_new_shuffle_id() -> int:
    """
    Get a new available shuffle ID.

    Since RapidsMPF only supports a limited number of shuffler instances at
    any given time, this function maintains a shared pool of shuffle IDs.

    If no IDs are available locally, it queries all workers for IDs in use,
    updates the vacancy set accordingly, and retries. If all IDs are in use
    across the cluster, an error is raised.

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

            def get_occupied_ids() -> set[int]:
                ctx = get_single_worker_context()
                with ctx.lock:
                    return set(ctx.shufflers.keys())

            # We start with setting all IDs as vacant and then subtract all
            # IDs occupied on any one worker.
            _shuffle_id_vacancy = set(range(Shuffler.max_concurrent_shuffles))
            _shuffle_id_vacancy.difference_update(get_occupied_ids())
            if not _shuffle_id_vacancy:
                raise ValueError(
                    f"Cannot manage moret than {Shuffler.max_concurrent_shuffles} "
                    "shuffles at once."
                )

        return _shuffle_id_vacancy.pop()


def get_single_shuffler(
    shuffle_id: int,
    partition_count: int | None = None,
) -> Shuffler:
    """
    Return the appropriate :class:`Shuffler` object.

    Parameters
    ----------
    shuffle_id
        Unique ID for the shuffle operation.
    partition_count
        Output partition count for the shuffle operation.

    Returns
    -------
    The active RapidsMPF :class:`Shuffler` object associated with
    the specified ``shuffle_id`` and ``partition_count`.

    Notes
    -----
    Whenever a new :class:`Shuffler` object is created, it is
    saved as ``WorkerContext.shufflers[shuffle_id]``.
    """
    ctx = get_single_worker_context()
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


def _single_worker_barrier(
    shuffle_ids: tuple[int, ...],
    partition_count: int,
    dependencies: Sequence[None],
) -> None:
    """
    Single worker barrier for RapidsMPF shuffle.

    Parameters
    ----------
    shuffle_ids
        Tuple of shuffle ids associated with the current
        task graph. This tuple will only contain a single
        integer when `single_rapidsmpf_shuffle_graph` is
        used for graph generation.
    partition_count
        Number of output partitions for the current shuffle.
    dependencies
        Null sequence used to enforce barrier dependencies.
    """
    for shuffle_id in shuffle_ids:
        shuffler = get_single_shuffler(shuffle_id)
        for pid in range(partition_count):
            shuffler.insert_finished(pid)


def _stage_single_shuffler(shuffle_id: int, partition_count: int) -> None:
    """
    Stage a shuffler object without returning it.

    Parameters
    ----------
    shuffle_id
        Unique ID for the shuffle operation.
    partition_count
        Output partition count for the shuffle operation.
    """
    get_single_shuffler(shuffle_id, partition_count=partition_count)


def _insert_partition_single(
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
    callback
        Insertion callback function. This function must be
        the `insert_partition` attribute of a `ShufflerIntegration`
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
    if callback is None:
        raise ValueError("callback missing in _insert_partition_single.")

    callback(
        df,
        partition_id,
        partition_count,
        get_single_shuffler(shuffle_id),
        options,
        *other_keys,
    )


def _extract_partition_single(
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
    callback
        Insertion callback function. This function must be
        the `extract_partition` attribute of a `ShufflerIntegration`
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
    shuffler = get_single_shuffler(shuffle_id)
    try:
        return callback(
            partition_id,
            shuffler,
            options,
        )
    finally:
        if shuffler.finished():
            ctx = get_single_worker_context()
            with ctx.lock:
                if shuffle_id in ctx.shufflers:
                    del ctx.shufflers[shuffle_id]


def single_rapidsmpf_shuffle_graph(
    input_name: str,
    output_name: str,
    partition_count_in: int,
    partition_count_out: int,
    integration: ShufflerIntegration,
    options: Any,
    *other_keys: str | tuple[str, int],
    config_options: Options = Options(),
) -> dict[Any, Any]:
    """
    Return the task graph for a RapidsMPF shuffle.

    Parameters
    ----------
    input_name
        The task name for input DataFrame tasks.
    output_name
        The task name for output DataFrame tasks.
    partition_count_in
        Partition count of input collection.
    partition_count_out
        Partition count of output collection.
    integration
        Shuffle-integration specification.
    options
        Optional key-word arguments.
    *other_keys
        Other keys needed by ``integration.insert_partition``.
    config_options
        RapidsMPF configuration options.

    Returns
    -------
    A valid task graph for single-worker execution.
    """
    # Make sure single worker is initialized
    setup_single_worker(config_options)

    # Get the shuffle id
    shuffle_id = _get_new_shuffle_id()
    _stage_single_shuffler(shuffle_id, partition_count_out)

    # Check integration argument
    if not isinstance(integration, ShufflerIntegration):
        raise TypeError(f"Expected ShufflerIntegration object, got {integration}.")

    # Define task names for each phase of the shuffle
    insert_name = f"rmpf-insert-{output_name}"
    worker_barrier_name = f"rmpf-worker-barrier-{output_name}"

    # Add tasks to insert each partition into the shuffler
    graph: dict[Any, Any] = {
        (insert_name, pid): (
            _insert_partition_single,
            integration.insert_partition,
            (input_name, pid),
            pid,
            partition_count_out,
            shuffle_id,
            options,
            *other_keys,
        )
        for pid in range(partition_count_in)
    }

    # Add global barrier task
    graph[(worker_barrier_name, 0)] = (
        _single_worker_barrier,
        (shuffle_id,),
        partition_count_out,
        list(graph.keys()),
    )

    # Add extraction tasks
    output_keys = []
    for part_id in range(partition_count_out):
        output_keys.append((output_name, part_id))
        graph[output_keys[-1]] = (
            _extract_partition_single,
            integration.extract_partition,
            shuffle_id,
            part_id,
            (worker_barrier_name, 0),
            options,
        )

    return graph
