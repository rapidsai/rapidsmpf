# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Join integration for Dask Distributed clusters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from rapidsmpf.config import Options
from rapidsmpf.integrations.core import join_partition
from rapidsmpf.integrations.dask.core import (
    get_dask_client,
    get_dask_worker_rank,
    get_worker_context,
)
from rapidsmpf.integrations.dask.shuffler import _shuffle_insertion_graph

if TYPE_CHECKING:
    from rapidsmpf.integrations.core import JoinIntegration


def rapidsmpf_join_graph(
    left_name: str,
    right_name: str,
    output_name: str,
    left_partition_count_in: int,
    right_partition_count_in: int,
    integration: JoinIntegration,
    left_options: Any,
    right_options: Any,
    join_options: Any,
    *,
    bcast_side: Literal["left", "right", None] = None,
    left_pre_shuffled: bool = False,
    right_pre_shuffled: bool = False,
    config_options: Options = Options(),
) -> dict[Any, Any]:
    """
    Return the task graph for a RapidsMPF join.

    Parameters
    ----------
    left_name
        The name of the left table.
    right_name
        The name of the right table.
    output_name
        The name of the output table.
    left_partition_count_in
        The number of partitions in the left table.
    right_partition_count_in
        The number of partitions in the right table.
    integration
        The JoinIntegration protocol to use.
    left_options
        Additional options for extracting the left table.
    right_options
        Additional options for extracting the right table.
    join_options
        Additional options for the join.
    bcast_side
        The side of the join being broadcasted.
        Options are ``{'left', 'right', None}``.
        Note: Only ``None`` is supported for now.
    left_pre_shuffled
        Whether the left table is already shuffled.
    right_pre_shuffled
        Whether the right table is already shuffled.
    config_options
        RapidsMPF configuration options.

    Returns
    -------
    The task graph for the join operation.
    """
    # Get the Dask client and worker ranks
    client = get_dask_client(options=config_options)
    worker_ranks: dict[int, str] = {
        v: k for k, v in client.run(get_dask_worker_rank).items()
    }
    n_workers = len(worker_ranks)

    # Build the task-graph and restricted-key dicts incrementally
    restricted_keys: dict[Any, str] = {}
    graph: dict[Any, Any] = {}
    left_barrier_name: str | None = None
    right_barrier_name: str | None = None
    left_op_id: int | None = None
    right_op_id: int | None = None

    if bcast_side is None:
        # Regular hash join

        # Determine the number of partitions in the output table
        partition_count_out = max(left_partition_count_in, right_partition_count_in)

        # Shuffle left side (if necessary)
        if not left_pre_shuffled or left_partition_count_in != partition_count_out:
            (
                left_graph,
                left_barrier_name,
                left_restricted_keys,
                left_op_id,
            ) = _shuffle_insertion_graph(
                client,
                left_name,
                f"left-{output_name}",
                left_partition_count_in,
                partition_count_out,
                integration.get_shuffler_integration(),
                worker_ranks,
                left_options,
            )
            restricted_keys.update(left_restricted_keys)
            graph.update(left_graph)

        # Shuffle right side (if necessary)
        if not right_pre_shuffled or right_partition_count_in != partition_count_out:
            (
                right_graph,
                right_barrier_name,
                right_restricted_keys,
                right_op_id,
            ) = _shuffle_insertion_graph(
                client,
                right_name,
                f"right-{output_name}",
                right_partition_count_in,
                partition_count_out,
                integration.get_shuffler_integration(),
                worker_ranks,
                right_options,
            )
            restricted_keys.update(right_restricted_keys)
            graph.update(right_graph)

        # Add basic hash-join tasks
        for part_id in range(partition_count_out):
            rank = part_id % n_workers
            n_worker_tasks = partition_count_out // n_workers + int(
                rank < (partition_count_out % n_workers)
            )
            key = (output_name, part_id)
            graph[key] = (
                join_partition,
                get_worker_context,
                integration,
                None,  # Not a broadcast join
                left_op_id,
                right_op_id,
                left_barrier_name or (left_name, part_id),
                right_barrier_name or (right_name, part_id),
                part_id,
                n_worker_tasks,
                left_options,
                right_options,
                join_options,
            )
            # Assume round-robin partition assignment
            restricted_keys[key] = worker_ranks[rank]

    elif bcast_side in ["left", "right"]:  # pragma: no cover
        # TODO: Broadcast join
        raise NotImplementedError("Broadcast join is not yet implemented.")
    else:  # pragma: no cover
        raise ValueError(f"Invalid broadcast side: {bcast_side}")

    # Tell the scheduler to restrict the worker-specific keys
    client._send_to_scheduler(
        {
            "op": "rmpf_add_restricted_tasks",
            "tasks": restricted_keys,
        }
    )

    # Return the full join task graph
    return graph
