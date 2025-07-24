# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffler integration with external libraries."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar, runtime_checkable

from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmpf.buffer.spill_collection import SpillCollection
from rapidsmpf.config import Options
from rapidsmpf.shuffler import Shuffler

if TYPE_CHECKING:
    from collections.abc import Callable

    from rapidsmpf.buffer.resource import BufferResource
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.progress_thread import ProgressThread
    from rapidsmpf.statistics import Statistics


DataFrameT = TypeVar("DataFrameT")


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
    spill_collection
        A collection of Python objects that can be spilled to free up device memory.
    statistics
        The statistics used by the worker. If None, statistics is disabled.
    shufflers
        A mapping from shuffler IDs to active shuffler instances.
    options
        Configuration options.
    """

    lock: ClassVar[threading.RLock] = threading.RLock()
    br: BufferResource | None = None
    progress_thread: ProgressThread | None = None
    comm: Communicator | None = None
    spill_collection: SpillCollection = field(default_factory=SpillCollection)
    statistics: Statistics | None = None
    shufflers: dict[int, Shuffler] = field(default_factory=dict)
    options: Options = field(default_factory=Options)


@runtime_checkable
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


def get_shuffler(
    get_context: Callable[..., WorkerContext],
    shuffle_id: int,
    *,
    partition_count: int | None = None,
    worker: Any = None,
) -> Shuffler:
    """
    Return the appropriate :class:`Shuffler` object.

    Parameters
    ----------
    get_context
        Callable function to fetch the worker context.
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
    ctx = get_context(worker)
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
        Insertion callback function. This function must be
        the `insert_partition` attribute of a `DaskIntegration`
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
        get_shuffler(get_context, shuffle_id),
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
        Insertion callback function. This function must be
        the `extract_partition` attribute of a `DaskIntegration`
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
    shuffler = get_shuffler(get_context, shuffle_id)
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
