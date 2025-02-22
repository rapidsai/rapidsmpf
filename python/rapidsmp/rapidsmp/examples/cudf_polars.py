# Copyright (c) 2025, NVIDIA CORPORATION.
"""Dask + cudf-Polars integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cudf_polars.containers import DataFrame

import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmp.examples.dask import (
    get_dask_client,
    get_shuffle_id,
    get_shuffler,
    get_worker_rank,
    global_rmp_barrier,
    stage_shuffler,
    worker_rmp_barrier,
)
from rapidsmp.shuffler import partition_and_pack, unpack_and_concat

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence


def rmp_shuffle_insert_polars(
    df: DataFrame,
    on: Sequence[str],
    partition_count: int,
    shuffle_id: int,
):
    """Add cudf-polars DataFrame chunks to an RMP shuffler."""
    shuffler = get_shuffler(shuffle_id, partition_count=partition_count)
    columns_to_hash = tuple(df.column_names.index(val) for val in on)
    packed_inputs = partition_and_pack(
        df.table,
        columns_to_hash=columns_to_hash,
        num_partitions=partition_count,
        stream=DEFAULT_STREAM,
        device_mr=rmm.mr.get_current_device_resource(),
    )
    shuffler.insert_chunks(packed_inputs)

    return shuffle_id


def rmp_shuffle_extract_polars(
    shuffle_id: int,
    partition_id: int,
    column_names: list[str],
    worker_barrier: tuple[int, ...],
):
    """Extract a finished partition from the RMP shuffler."""
    shuffler = get_shuffler(shuffle_id)
    shuffler.wait_on(partition_id)
    return DataFrame.from_table(
        unpack_and_concat(
            shuffler.extract(partition_id),
            stream=DEFAULT_STREAM,
            device_mr=rmm.mr.get_current_device_resource(),
        ),
        column_names,
    )


def make_rmp_shuffle_graph(
    input_name: str,
    output_name: str,
    column_names: Sequence[str],
    shuffle_on: Sequence[str],
    partition_count_in: int,
    partition_count_out: int,
) -> MutableMapping[Any, Any]:
    """Make a rapidsmp shuffle task graph."""
    # Get the shuffle id
    client = get_dask_client()
    shuffle_id = get_shuffle_id()

    # Extract mapping between ranks and worker addresses
    worker_ranks: dict[int, str] = {
        v: k for k, v in client.run(get_worker_rank).items()
    }
    n_workers = len(worker_ranks)
    restricted_keys: MutableMapping[Any, str] = {}

    # Define task names for each stage of the shuffle
    insert_name = f"rmp-insert-{output_name}"
    global_barrier_name = f"rmp-global-barrier-{output_name}"
    worker_barrier_name = f"rmp-worker-barrier-{output_name}"

    # Stage a shuffler on every worker for this shuffle id
    client.run(
        stage_shuffler,
        shuffle_id=shuffle_id,
        partition_count=partition_count_out,
    )

    # Add operation to submit each partition to the shuffler
    dsk: MutableMapping[Any, Any] = {
        (insert_name, pid): (
            rmp_shuffle_insert_polars,
            (input_name, pid),
            shuffle_on,
            partition_count_out,
            shuffle_id,
        )
        for pid in range(partition_count_in)
    }

    # Add global barrier task
    dsk[(global_barrier_name, 0)] = (
        global_rmp_barrier,
        (shuffle_id,),
        list(dsk.keys()),
    )

    # Add worker barrier tasks
    worker_barriers: MutableMapping[Any, Any] = {}
    for rank, addr in worker_ranks.items():
        key = (worker_barrier_name, rank)
        worker_barriers[rank] = key
        dsk[key] = (
            worker_rmp_barrier,
            (shuffle_id,),
            partition_count_out,
            (global_barrier_name, 0),
        )
        restricted_keys[key] = addr

    # Add extraction tasks
    output_keys = []
    for part_id in range(partition_count_out):
        rank = part_id % n_workers
        output_keys.append((output_name, part_id))
        dsk[output_keys[-1]] = (
            rmp_shuffle_extract_polars,
            shuffle_id,
            part_id,
            column_names,
            worker_barriers[rank],
        )
        # Assume round-robin partition assignment
        restricted_keys[output_keys[-1]] = worker_ranks[rank]

    # Tell the scheduler to restrict the shuffle keys
    # to specific workers
    client._send_to_scheduler(
        {
            "op": "rmp_add_restricted_tasks",
            "tasks": restricted_keys,
        }
    )

    # Return the graph
    return dsk
