# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Join integration for Dask Distributed clusters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from rapidsmpf.config import Options
from rapidsmpf.integrations.core import join_chunk
from rapidsmpf.integrations.dask.core import (
    get_dask_client,
    get_dask_worker_rank,
    get_worker_context,
)
from rapidsmpf.integrations.dask.shuffler import _partial_shuffle_graph

if TYPE_CHECKING:
    from rapidsmpf.integrations.core import JoinIntegration


def rapidsmpf_join_graph(
    left_name: str,
    right_name: str,
    output_name: str,
    bcast_side: Literal["left", "right", "none"],
    left_partition_count_in: int,
    right_partition_count_in: int,
    integration: JoinIntegration,
    options: Any,
    *,
    config_options: Options = Options(),
) -> dict[Any, Any]:
    """Return the task graph for a RapidsMPF broadcast join."""
    client = get_dask_client(options=config_options)
    worker_ranks: dict[int, str] = {
        v: k for k, v in client.run(get_dask_worker_rank).items()
    }
    n_workers = len(worker_ranks)
    restricted_keys: dict[Any, str] = {}
    graph: dict[Any, Any] = {}

    if bcast_side == "none":
        # Regular hash join
        partition_count_out = max(left_partition_count_in, right_partition_count_in)
        # TODO: What if one or both sides is already shuffled?
        # Perhaps the user shouldn't use this function in that
        # case, but we may be able to handle it here.

        # Shuffle left side
        left_barrier_name = f"rmpf-shuffle-left-{output_name}"
        left_op_id, left_restricted_keys, left_graph = _partial_shuffle_graph(
            client,
            left_name,
            left_barrier_name,
            left_partition_count_in,
            partition_count_out,
            integration.shuffler_integration(),
            worker_ranks,
            {"on": options["left_on"]},
        )
        restricted_keys.update(left_restricted_keys)
        graph.update(left_graph)

        # Shuffle right side
        right_barrier_name = f"rmpf-shuffle-right-{output_name}"
        right_op_id, right_restricted_keys, right_graph = _partial_shuffle_graph(
            client,
            right_name,
            right_barrier_name,
            right_partition_count_in,
            partition_count_out,
            integration.shuffler_integration(),
            worker_ranks,
            {"on": options["right_on"]},
        )
        restricted_keys.update(right_restricted_keys)
        graph.update(right_graph)

        # Add basic hash-join tasks
        for part_id in range(partition_count_out):
            rank = part_id % n_workers
            n_worker_chunks = partition_count_out // n_workers + int(
                rank < (partition_count_out % n_workers)
            )
            key = (output_name, part_id)
            graph[key] = (
                join_chunk,
                get_worker_context,
                integration.join_chunk,
                bcast_side,
                left_op_id,
                right_op_id,
                left_barrier_name,
                right_barrier_name,
                part_id,
                n_worker_chunks,
                options,
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

    return graph
