# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffler integration for Dask Distributed clusters."""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, Any

from distributed import get_worker

from rapidsmpf.config import Options
from rapidsmpf.integrations.core import (
    extract_partition,
    get_new_shuffle_id,
    get_shuffler,
    insert_partition,
)
from rapidsmpf.integrations.dask.core import (
    get_dask_client,
    get_dask_worker_rank,
    get_worker_context,
    global_rmpf_barrier,
)

if TYPE_CHECKING:
    from numbers import Number

    from distributed import Client, Worker

    from rapidsmpf.integrations.core import ShufflerIntegration


def _get_occupied_ids_local(dask_worker: Worker) -> set[int]:
    ctx = get_worker_context(dask_worker)
    with ctx.lock:
        return set(ctx.shufflers.keys())


def _get_occupied_ids_dask(client: Client) -> list[set[int]]:
    return list(client.run(_get_occupied_ids_local).values())


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
    for shuffle_id in shuffle_ids:
        shuffler = get_shuffler(get_worker_context(), shuffle_id)
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
        get_worker_context(dask_worker),
        shuffle_id,
        partition_count=partition_count,
        worker=dask_worker,
    )


def _get_dask_worker_ranks_and_stage_shuffler(
    shuffle_id: int, partition_count: int, dask_worker: Worker | None = None
) -> int:
    rank = get_dask_worker_rank(dask_worker)

    _stage_shuffler(shuffle_id, partition_count, dask_worker)

    return rank


def _shuffle_insertion_graph(
    client: Client,
    input_name: str,
    output_name_root: str,
    partition_count_in: int,
    partition_count_out: int,
    integration: ShufflerIntegration,
    worker_ranks: dict[int, str],
    options: Any,
    *,
    other_keys: tuple[str | tuple[str, int], ...] = (),
    shuffle_id: int | None = None,
) -> tuple[dict[Any, Any], str, dict[Any, str], int]:
    """
    Return the insertion task graph for a RapidsMPF shuffle.

    Parameters
    ----------
    client
        The Dask client.
    input_name
        The task name for input DataFrame tasks.
    output_name_root
        The root name to used for new tasks in the
        generated graph.
    partition_count_in
        Partition count of input collection.
    partition_count_out
        Partition count of output collection.
    integration
        Dask-integration specification.
    worker_ranks
        A dictionary of known worker ranks and addresses.
    options
        Optional key-word arguments.
    other_keys
        Other keys needed by ``integration.insert_partition``.
    shuffle_id
        The shuffle id to use. If not provided, a new shuffler
        will be staged.

    Returns
    -------
    graph
        The shuffle-insertion task graph.
    output_barrier_name
        The name of the output shuffle-barrier task.
    restricted_keys
        The restricted keys for the generated task graph.
    shuffle_id
        The shuffle id used in the generated task graph.
        This will be the same as the input shuffle id if one was provided.

    Notes
    -----
    This function is used to build the partial task graph needed
    to shuffle a single table without extracting the partitions.
    """
    # Get the shuffle id and worker ranks
    if shuffle_id is None:
        # Need a new shuffler
        shuffle_id = get_new_shuffle_id(partial(_get_occupied_ids_dask, client))
        client.run(_stage_shuffler, shuffle_id, partition_count_out)

    # Define task names for each phase of the shuffle
    insert_name = f"rmpf-insert-{output_name_root}"
    global_barrier_name = f"rmpf-global-barrier-1-{output_name_root}"
    worker_barrier_name = f"rmpf-worker-barrier-{output_name_root}"
    output_barrier_name = f"rmpf-global-barrier-2-{output_name_root}"

    # Add tasks to insert each partition into the shuffler
    graph: dict[Any, Any] = {
        (insert_name, pid): (
            insert_partition,
            get_worker_context,
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
    graph[global_barrier_name] = (
        global_rmpf_barrier,
        *graph.keys(),
    )

    # Add worker barrier tasks
    worker_barriers: dict[Any, Any] = {}
    restricted_keys: dict[Any, str] = {}
    for rank, addr in worker_ranks.items():
        key = (worker_barrier_name, rank)
        worker_barriers[rank] = key
        graph[key] = (
            _worker_rmpf_barrier,
            (shuffle_id,),
            partition_count_out,
            global_barrier_name,
        )
        restricted_keys[key] = addr

    # Add global barrier task
    graph[output_barrier_name] = (
        global_rmpf_barrier,
        *worker_barriers.values(),
    )

    return graph, output_barrier_name, restricted_keys, shuffle_id


def rapidsmpf_shuffle_graph(
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
        Dask-integration specification.
    options
        Optional key-word arguments.
    *other_keys
        Other keys needed by ``integration.insert_partition``.
    config_options
        RapidsMPF configuration options.

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
    and `rapidsmpf.integrations.cudf.partition.unpack_and_concat`).

    The extraction phase will include a single task for each of
    the ``partition_count_out`` partitions in the shuffled output
    DataFrame. The extraction logic must be defined by the
    ``extract_partition`` classmethod of the ``integration`` argument.

    Extraction tasks must be restricted to specific Dask workers,
    and they must also depend on the second global-barrier task.
    """
    # Get the shuffle id
    client = get_dask_client(options=config_options)
    shuffle_id = get_new_shuffle_id(partial(_get_occupied_ids_dask, client))

    # Note: We've observed high overhead from `Client.run` on some systems with
    # some networking configurations. Minimize the number of `Client.run` calls
    # by batching as much work as possible into a single call as possible.
    # See https://github.com/rapidsai/rapidsmpf/pull/323 for more.
    worker_ranks: dict[int, str] = {
        v: k
        for k, v in client.run(
            _get_dask_worker_ranks_and_stage_shuffler,
            shuffle_id,
            partition_count_out,
        ).items()
    }

    # Generate the shuffle-insertion graph.
    # The same partial-graph logic is used for joins.
    graph, shuffled_name, restricted_keys, _ = _shuffle_insertion_graph(
        client,
        input_name,
        output_name,
        partition_count_in,
        partition_count_out,
        integration,
        worker_ranks,
        options,
        other_keys=other_keys,
        shuffle_id=shuffle_id,
    )

    # Add extraction tasks
    output_keys = []
    n_workers = len(worker_ranks)
    for part_id in range(partition_count_out):
        rank = part_id % n_workers
        output_keys.append((output_name, part_id))
        graph[output_keys[-1]] = (
            extract_partition,
            get_worker_context,
            integration.extract_partition,
            shuffle_id,
            part_id,
            shuffled_name,
            options,
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


def _gather_worker_shuffle_statistics(
    dask_worker: Worker,
) -> dict[str, dict[str, Number]]:
    context = get_worker_context(dask_worker)
    return context.get_statistics()


def _clear_worker_shuffle_statistics(
    dask_worker: Worker,
) -> None:
    context = get_worker_context(dask_worker)
    context.statistics.clear()


def clear_shuffle_statistics(client: Client) -> None:
    """
    Clear all statistics for all workers.

    Memory profiling records are not cleared.

    Parameters
    ----------
    client
        The Dask client.
    """
    client.run(_clear_worker_shuffle_statistics)


def gather_shuffle_statistics(client: Client) -> dict[str, dict[str, int | float]]:
    """
    Gather shuffle statistics from all workers.

    Parameters
    ----------
    client
        The Dask client.

    Returns
    -------
    A dictionary of statistics. The keys are the names of each statistic. The values
    are a dictionary with two keys:

        - "count" is the number of times the statistic was recorded (summed
          across all workers).
        - "value" is value of the statistic (summed across all workers).

    Notes
    -----
    Statistics are global across all shuffles. To measure statistics for any
    given shuffle, you can clear the accumulated statistics between runs
    with :func:`clear_shuffle_statistics`.
    """
    # {address: {stat: {count: int, value: int}}}
    # collect
    stats: dict[str, dict[str, dict[str, Number]]] = client.run(
        _gather_worker_shuffle_statistics
    )  # type: ignore[arg-type]
    # aggregate
    result: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {"count": 0, "value": 0.0}
    )

    for worker_stats in stats.values():
        # the types are a bit fiddly here. We say they're "Number", but really
        # we know that counts are ints and values are floats
        for name, stat in worker_stats.items():
            result[name]["count"] += stat["count"]  # type: ignore[operator]
            result[name]["value"] += stat["value"]  # type: ignore[operator]

    return dict(result)
