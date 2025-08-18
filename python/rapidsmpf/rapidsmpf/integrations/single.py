# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffler integration for single-worker pylibcudf execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rapidsmpf.communicator.single import new_communicator
from rapidsmpf.config import Options
from rapidsmpf.integrations.core import (
    WorkerContext,
    extract_partition,
    get_new_shuffle_id,
    get_shuffler,
    insert_partition,
    rmpf_worker_setup,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rapidsmpf.integrations.core import ShufflerIntegration


_worker_context: WorkerContext | None = None


def get_worker_context() -> WorkerContext:
    """
    Retrieve the single-worker ``WorkerContext``.

    Returns
    -------
    The worker context

    Raises
    ------
    ValueError
        If a worker context was never created.

    See Also
    --------
    setup_worker
        Must be called before this function.
    """
    with WorkerContext.lock:
        if _worker_context is None:
            raise RuntimeError("Must call setup_worker first")
        return _worker_context


def setup_worker(options: Options = Options()) -> None:
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
    global _worker_context  # noqa: PLW0603
    with WorkerContext.lock:
        if _worker_context is None:
            comm = new_communicator(options)
            _worker_context = rmpf_worker_setup(
                None, "single_", comm=comm, options=options
            )


def _get_occupied_ids() -> list[set[int]]:
    ctx = get_worker_context()
    return [set(ctx.shufflers.keys())]


def _barrier(
    shuffle_ids: tuple[int, ...],
    partition_count: int,
    *dependencies: Sequence[None],
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
        shuffler = get_shuffler(get_worker_context(), shuffle_id)
        for pid in range(partition_count):
            shuffler.insert_finished(pid)


def _stage_shuffle(shuffle_id: int, partition_count: int) -> None:
    """
    Stage a shuffler object without returning it.

    Parameters
    ----------
    shuffle_id
        Unique ID for the shuffle operation.
    partition_count
        Output partition count for the shuffle operation.
    """
    get_shuffler(
        get_worker_context(),
        shuffle_id,
        partition_count=partition_count,
    )


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

    This shuffle will use the single-process RapidsMPF Communicator.

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
    setup_worker(config_options)

    # Get the shuffle id
    shuffle_id = get_new_shuffle_id(_get_occupied_ids)
    _stage_shuffle(shuffle_id, partition_count_out)

    # Define task names for each phase of the shuffle
    insert_name = f"rmpf-insert-{output_name}"
    worker_barrier_name = f"rmpf-worker-barrier-{output_name}"

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
    graph[(worker_barrier_name, 0)] = (
        _barrier,
        (shuffle_id,),
        partition_count_out,
        *graph.keys(),
    )

    # Add extraction tasks
    output_keys = []
    for part_id in range(partition_count_out):
        output_keys.append((output_name, part_id))
        graph[output_keys[-1]] = (
            extract_partition,
            get_worker_context,
            integration.extract_partition,
            shuffle_id,
            part_id,
            (worker_barrier_name, 0),
            options,
        )

    return graph
