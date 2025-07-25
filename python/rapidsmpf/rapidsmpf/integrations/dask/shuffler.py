# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffler integration for Dask Distributed clusters."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from distributed import get_worker

from rapidsmpf.config import Options
from rapidsmpf.integrations.core import (
    extract_partition,
    get_shuffler,
    insert_partition,
)
from rapidsmpf.integrations.dask.core import (
    get_dask_client,
    get_dask_worker_context,
    get_dask_worker_rank,
    global_rmpf_barrier,
)
from rapidsmpf.shuffler import Shuffler

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from distributed import Client, Worker

    from rapidsmpf.integrations.core import ShufflerIntegration


# Set of available shuffle IDs
_shuffle_id_vacancy: set[int] = set(range(Shuffler.max_concurrent_shuffles))
_shuffle_id_vacancy_lock: threading.Lock = threading.Lock()


def _get_new_shuffle_id(client: Client) -> int:
    """
    Get a new available shuffle ID.

    Since RapidsMPF only supports a limited number of shuffler instances at
    any given time, this function maintains a shared pool of shuffle IDs.

    If no IDs are available locally, it queries all workers for IDs in use,
    updates the vacancy set accordingly, and retries. If all IDs are in use
    across the cluster, an error is raised.

    Parameters
    ----------
    client
        A Dask distributed client used to query workers for active shuffle IDs.

    Returns
    -------
    A unique shuffle ID not currently in use.

    Raises
    ------
    ValueError
        If all shuffle IDs are currently in use across the cluster.
    """
    global _shuffle_id_vacancy  # noqa: PLW0603

    with _shuffle_id_vacancy_lock:
        if not _shuffle_id_vacancy:

            def get_occupied_ids(dask_worker: Worker) -> set[int]:
                ctx = get_dask_worker_context(dask_worker)
                with ctx.lock:
                    return set(ctx.shufflers.keys())

            # We start with setting all IDs as vacant and then subtract all
            # IDs occupied on any one worker.
            _shuffle_id_vacancy = set(range(Shuffler.max_concurrent_shuffles))
            _shuffle_id_vacancy.difference_update(
                *client.run(get_occupied_ids).values()
            )
            if not _shuffle_id_vacancy:
                raise ValueError(
                    f"Cannot shuffle more than {Shuffler.max_concurrent_shuffles} "
                    "times in a single Dask compute."
                )

        return _shuffle_id_vacancy.pop()


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
    # from rapidsmpf.integrations.dask.shuffler import get_shuffler
    for shuffle_id in shuffle_ids:
        shuffler = get_shuffler(get_dask_worker_context, shuffle_id)
        for pid in range(partition_count):
            shuffler.insert_finished(pid)


def _stage_shuffler(
    shuffle_id: int,
    partition_count: int,
    worker: Worker | None = None,
) -> None:
    """
    Stage a shuffler object without returning it.

    Parameters
    ----------
    shuffle_id
        Unique ID for the shuffle operation.
    partition_count
        Output partition count for the shuffle operation.
    worker
        The current dask worker.

    Notes
    -----
    This function is expected to run on a Dask worker.
    """
    worker = worker or get_worker()
    get_shuffler(
        get_dask_worker_context,
        shuffle_id,
        partition_count=partition_count,
        worker=worker,
    )


def _get_dask_worker_ranks_and_stage_shuffler(
    shuffle_id: int, partition_count: int, dask_worker: Worker | None = None
) -> int:
    rank = get_dask_worker_rank(dask_worker)

    _stage_shuffler(shuffle_id, partition_count, dask_worker)

    return rank


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
    shuffle_id = _get_new_shuffle_id(client)

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

    n_workers = len(worker_ranks)
    restricted_keys: MutableMapping[Any, str] = {}

    # Define task names for each phase of the shuffle
    insert_name = f"rmpf-insert-{output_name}"
    global_barrier_1_name = f"rmpf-global-barrier-1-{output_name}"
    global_barrier_2_name = f"rmpf-global-barrier-2-{output_name}"
    worker_barrier_name = f"rmpf-worker-barrier-{output_name}"

    # Add tasks to insert each partition into the shuffler
    graph: dict[Any, Any] = {
        (insert_name, pid): (
            insert_partition,
            get_dask_worker_context,
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
            extract_partition,
            get_dask_worker_context,
            integration.extract_partition,
            shuffle_id,
            part_id,
            (global_barrier_2_name, 0),
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
