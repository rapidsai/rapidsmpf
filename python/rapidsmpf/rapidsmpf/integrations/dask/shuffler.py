# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffler integration for Dask Distributed clusters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from distributed import get_worker

from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmpf.integrations.dask.core import (
    DataFrameT,
    get_dask_client,
    get_worker_context,
    get_worker_rank,
    global_rmpf_barrier,
)
from rapidsmpf.shuffler import Shuffler

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping, Sequence

    from distributed import Worker


_shuffle_counter: int = 0


def get_shuffle_id() -> int:
    """
    Return the unique id for a new shuffle.

    Returns
    -------
    The enumerated integer id for the current shuffle.
    """
    global _shuffle_counter  # noqa: PLW0603

    _shuffle_counter += 1
    return _shuffle_counter


def get_shuffler(
    shuffle_id: int,
    partition_count: int | None = None,
    dask_worker: Worker | None = None,
) -> Shuffler:
    """
    Return the appropriate :class:`Shuffler` object.

    Parameters
    ----------
    shuffle_id
        Unique ID for the shuffle operation.
    partition_count
        Output partition count for the shuffle operation.
    dask_worker
        The current dask worker.

    Returns
    -------
    The active RapidsMPF :class:`Shuffler` object associated with
    the specified ``shuffle_id``, ``partition_count`` and
    ``dask_worker``.

    Notes
    -----
    Whenever a new :class:`Shuffler` object is created, it is
    saved as ``DaskWorkerContext.shufflers[shuffle_id]``.

    This function is expected to run on a Dask worker.
    """
    ctx = get_worker_context(dask_worker)
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


@runtime_checkable
class DaskIntegration(Protocol[DataFrameT]):
    """
    dask-integration protocol.

    This protocol can be used to implement a RapidsMPF-shuffle
    operation using a Dask task graph.
    """

    @staticmethod
    def insert_partition(
        df: DataFrameT,
        on: Sequence[str],
        partition_count: int,
        shuffler: Shuffler,
    ) -> None:
        """
        Add a partition to a RapidsMPF Shuffler.

        Parameters
        ----------
        df
            DataFrame partition to add to a RapidsMPF shuffler.
        on
            Sequence of column names to shuffle on.
        partition_count
            Number of output partitions for the current shuffle.
        shuffler
            The RapidsMPF Shuffler object to extract from.
        """

    @staticmethod
    def extract_partition(
        partition_id: int,
        column_names: list[str],
        shuffler: Shuffler,
    ) -> DataFrameT:
        """
        Extract a DataFrame partition from a RapidsMPF Shuffler.

        Parameters
        ----------
        partition_id
            Partition id to extract.
        column_names
            Sequence of output column names.
        shuffler
            The RapidsMPF Shuffler object to extract from.

        Returns
        -------
        A shuffled DataFrame partition.
        """


def _worker_rmpf_barrier(
    shuffle_ids: tuple[int, ...],
    partition_count: int,
    dependency: None,
) -> None:
    """
    Worker barrier for RapidsMPF shuffle.

    Parameters
    ----------
    shuffle_ids
        Tuple of shuffle ids associated with the current
        task graph. This tuple will only contain a single
        integer when `rapidsmpf_shuffle_graph` is used for
        graph generation.
    partition_count
        Number of output partitions for the current shuffle.
    dependency
        Null argument used to enforce barrier dependencies.

    Notes
    -----
    A worker barrier task DOES need to be restricted
    to a specific Dask worker.
    """
    from rapidsmpf.integrations.dask.shuffler import get_shuffler

    for shuffle_id in shuffle_ids:
        shuffler = get_shuffler(shuffle_id)
        for pid in range(partition_count):
            shuffler.insert_finished(pid)


def _stage_shuffler(
    shuffle_id: int,
    partition_count: int,
    dask_worker: Worker | None = None,
) -> None:
    """
    Stage a shuffler object without returning it.

    Parameters
    ----------
    shuffle_id
        Unique ID for the shuffle operation.
    partition_count
        Output partition count for the shuffle operation.
    dask_worker
        The current dask worker.

    Notes
    -----
    This function is expected to run on a Dask worker.
    """
    dask_worker = dask_worker or get_worker()
    get_shuffler(
        shuffle_id,
        partition_count=partition_count,
        dask_worker=dask_worker,
    )


def _insert_partition(
    callback: Callable[[DataFrameT, Sequence[str], int, Shuffler], None],
    df: DataFrameT,
    on: Sequence[str],
    partition_count: int,
    shuffle_id: int,
) -> None:
    """
    Add a partition to a RapidsMPF Shuffler.

    Parameters
    ----------
    callback
        Insertion callback function. This function must be
        the `insert_partition` attribute of a `DaskIntegration`
        protocol.
    df
        DataFrame partition to add to a RapidsMPF shuffler.
    on
        Sequence of column names to shuffle on.
    partition_count
        Number of output partitions for the current shuffle.
    shuffle_id
        The RapidsMPF shuffle id.
    """
    if callback is None:
        raise ValueError("callback missing in _insert_partition.")
    from rapidsmpf.integrations.dask.shuffler import get_shuffler

    callback(
        df,
        on,
        partition_count,
        get_shuffler(shuffle_id),
    )


def _extract_partition(
    callback: Callable[[int, Sequence[str], Shuffler], DataFrameT],
    shuffle_id: int,
    partition_id: int,
    column_names: list[str],
    worker_barrier: tuple[int, ...],
) -> DataFrameT:
    """
    Extract a partition from a RapidsMPF Shuffler.

    Parameters
    ----------
    callback
        Insertion callback function. This function must be
        the `extract_partition` attribute of a `DaskIntegration`
        protocol.
    shuffle_id
        The RapidsMPF shuffle id.
    partition_id
        Partition id to extract.
    column_names
        Sequence of output column names.
    worker_barrier
        Worker-barrier task dependency. This value should
        not be used for compute logic.

    Returns
    -------
    Extracted DataFrame partition.
    """
    if callback is None:
        raise ValueError("Missing callback in _extract_partition.")
    return callback(
        partition_id,
        column_names,
        get_shuffler(shuffle_id),
    )


def rapidsmpf_shuffle_graph(
    input_name: str,
    output_name: str,
    column_names: Sequence[str],
    shuffle_on: Sequence[str],
    partition_count_in: int,
    partition_count_out: int,
    integration: DaskIntegration,
) -> dict[Any, Any]:
    """
    Return the task graph for a RapidsMPF shuffle.

    Parameters
    ----------
    input_name
        The task name for input DataFrame tasks.
    output_name
        The task name for output DataFrame tasks.
    column_names
        Sequence of output column names.
    shuffle_on
        Sequence of column names to shuffle on (by hash).
    partition_count_in
        Partition count of input collection.
    partition_count_out
        Partition count of output collection.
    integration
        Dask-integration specification.

    Returns
    -------
    A valid task graph for Dask execution.

    Notes
    -----
    A RapidsMPF shuffle operation comprises four general phases:

    **Staging phase**
    A new :class:`rapidsmpf.shuffler.Shuffler` object must be staged on every worker
    in the current Dask cluster.

    **Insertion phase**
    Each input partition is split into a dictionary of chunks,
    and that dictionary is passed to the appropriate :class:`rapidsmpf.shuffler.Shuffler`
    object (using `rapidsmpf.shuffler.Shuffler.insert_chunks`).

    The insertion phase will include a single task for each of
    the ``partition_count_in`` partitions in the input DataFrame.
    The partitioning and insertion logic must be defined by the
    ``insert_partition`` classmethod of the ``integration`` argument.

    Insertion tasks are NOT restricted to specific Dask workers.
    These tasks may run anywhere in the cluster.

    **Barrier phase**
    All :class:`rapidsmpf.shuffler.Shuffler` objects must be 'informed' that the insertion
    phase is complete (on all workers) before the subsequent
    extraction phase begins. We call this synchronization step
    the 'barrier phase'.

    The barrier phase comprises three types of barrier tasks:

    1. First global barrier - A single barrier task is used to
    signal that all input partitions have been submitted to
    a :class:`rapidsmpf.shuffler.Shuffler` object on one of the workers. This task may
    also run anywhere on the cluster, but it must depend on
    ALL insertion tasks.

    2. Worker barrier(s) - Each worker must execute a single
    worker-barrier task. This task will call `insert_finished`
    for every output partition on the local :class:`rapidsmpf.shuffler.Shuffler`. These
    tasks must be restricted to specific workers, and they
    must all depend on the first global barrier.

    3. Second global barrier - A single barrier task is used
    to signal that all workers are ready to begin the extraction
    phase. This task may run anywhere on the cluster, but it must
    depend on all worker-barrier tasks.

    **Extraction phase**
    Each output partition is extracted from the local
    :class:`rapidsmpf.shuffler.Shuffler` object on the worker (using `rapidsmpf.shuffler.Shuffler.wait_on`
    and `rapidsmpf.shuffler.unpack_and_concat`).

    The extraction phase will include a single task for each of
    the ``partition_count_out`` partitions in the shuffled output
    DataFrame. The extraction logic must be defined by the
    ``extract_partition`` classmethod of the ``integration`` argument.

    Extraction tasks must be restricted to specific Dask workers,
    and they must also depend on the second global-barrier task.
    """
    # Get the shuffle id
    client = get_dask_client()
    shuffle_id = get_shuffle_id()

    # Check integration argument
    if not isinstance(integration, DaskIntegration):
        raise TypeError(f"Expected DaskIntegration object, got {integration}.")

    # Extract mapping between ranks and worker addresses
    worker_ranks: dict[int, str] = {
        v: k for k, v in client.run(get_worker_rank).items()
    }
    n_workers = len(worker_ranks)
    restricted_keys: MutableMapping[Any, str] = {}

    # Define task names for each phase of the shuffle
    insert_name = f"rmpf-insert-{output_name}"
    global_barrier_1_name = f"rmpf-global-barrier-1-{output_name}"
    global_barrier_2_name = f"rmpf-global-barrier-2-{output_name}"
    worker_barrier_name = f"rmpf-worker-barrier-{output_name}"

    # Stage a shuffler on every worker for this shuffle id
    client.run(
        _stage_shuffler,
        shuffle_id=shuffle_id,
        partition_count=partition_count_out,
    )

    # Make sure shuffle_on does not contain duplicate keys
    if len(set(shuffle_on)) != len(shuffle_on):
        raise ValueError(f"Got duplicate keys in shuffle_on: {shuffle_on}")

    # Add tasks to insert each partition into the shuffler
    graph: dict[Any, Any] = {
        (insert_name, pid): (
            _insert_partition,
            integration.insert_partition,
            (input_name, pid),
            shuffle_on,
            partition_count_out,
            shuffle_id,
        )
        for pid in range(partition_count_in)
    }

    # Add global barrier task
    graph[(global_barrier_1_name, 0)] = (
        global_rmpf_barrier,
        list(graph.keys()),
    )

    # Add worker barrier tasks
    worker_barriers: dict[Any, Any] = {}
    for rank, addr in worker_ranks.items():
        key = (worker_barrier_name, rank)
        worker_barriers[rank] = key
        graph[key] = (
            _worker_rmpf_barrier,
            (shuffle_id,),
            partition_count_out,
            (global_barrier_1_name, 0),
        )
        restricted_keys[key] = addr

    # Add global barrier task
    graph[(global_barrier_2_name, 0)] = (
        global_rmpf_barrier,
        list(worker_barriers.values()),
    )

    # Add extraction tasks
    output_keys = []
    for part_id in range(partition_count_out):
        rank = part_id % n_workers
        output_keys.append((output_name, part_id))
        graph[output_keys[-1]] = (
            _extract_partition,
            integration.extract_partition,
            shuffle_id,
            part_id,
            column_names,
            (global_barrier_2_name, 0),
        )
        # Assume round-robin partition assignment
        restricted_keys[output_keys[-1]] = worker_ranks[rank]

    # Tell the scheduler to restrict the shuffle keys
    # to specific workers
    client._send_to_scheduler(
        {
            "op": "rmpf_add_restricted_tasks",
            "tasks": restricted_keys,
        }
    )

    return graph
