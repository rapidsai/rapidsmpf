# Copyright (c) 2025, NVIDIA CORPORATION.
"""Dask-cuDF integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dask.dataframe as dd
import pandas as pd
from dask.tokenize import tokenize

import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmp.examples.dask import (
    get_dask_client,
    get_memory_resource,
    get_shuffle_id,
    get_shuffler,
    get_worker_rank,
    global_rmp_barrier,
    worker_rmp_barrier,
)
from rapidsmp.shuffler import partition_and_pack, unpack_and_concat
from rapidsmp.testing import pylibcudf_to_cudf_dataframe

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    import cudf


def rmp_shuffle_insert(
    df: cudf.DataFrame,
    on: Sequence[str],
    partition_count: int,
    shuffle_id: int,
):
    """Add chunks to an RMP shuffler."""
    shuffler = get_shuffler(shuffle_id, partition_count=partition_count)

    columns_to_hash = tuple(list(df.columns).index(val) for val in on)
    packed_inputs = partition_and_pack(
        df.to_pylibcudf()[0],
        columns_to_hash=columns_to_hash,
        num_partitions=partition_count,
        stream=DEFAULT_STREAM,
        device_mr=rmm.mr.get_current_device_resource(),
    )
    shuffler.insert_chunks(packed_inputs)

    # Pass back a pd.DataFrame object to make this
    # a proper Dask-DataFrame collection (for now)
    return pd.DataFrame({"id": [shuffle_id]})


def rmp_shuffle_extract(
    shuffle_id: int,
    partition_id: int,
    column_names: list[str],
    worker_barrier: tuple[int, ...],
):
    """Extract a finished partition from the RMP shuffler."""
    shuffler = get_shuffler(shuffle_id)
    shuffler.wait_on(partition_id)
    table = unpack_and_concat(
        shuffler.extract(partition_id),
        stream=DEFAULT_STREAM,
        device_mr=rmm.mr.get_current_device_resource(),
    )
    return pylibcudf_to_cudf_dataframe(
        table,
        column_names=column_names,
    )


def rmp_merge_insert(
    left: cudf.DataFrame,
    right: cudf.DataFrame,
    left_on: Sequence[str],
    right_on: Sequence[str],
    left_shuffle_id: int,
    right_shuffle_id: int,
    partition_count: int,
):
    """Add chunkss to RMP shufflers."""
    for df, on, shuffle_id in [
        (left, left_on, left_shuffle_id),
        (right, right_on, right_shuffle_id),
    ]:
        shuffler = get_shuffler(shuffle_id, partition_count=partition_count)

        columns_to_hash = tuple(list(df.columns).index(val) for val in on)
        packed_inputs = partition_and_pack(
            df.to_pylibcudf()[0],
            columns_to_hash=columns_to_hash,
            num_partitions=partition_count,
            stream=DEFAULT_STREAM,
            device_mr=rmm.mr.get_current_device_resource(),
        )
        shuffler.insert_chunks(packed_inputs)

    # Pass back a pd.DataFrame object to make this
    # a proper Dask-DataFrame collection (for now)
    return pd.DataFrame({"id": [shuffle_id]})


def rmp_merge_extract(
    left_shuffle_id: int,
    right_shuffle_id: int,
    partition_id: int,
    left_column_names: list[str],
    right_column_names: list[str],
    merge_kwargs: dict,
    worker_barrier: tuple[int, ...],
):
    """Extract a finished partition from the RMP shuffler."""
    mr = get_memory_resource()
    rmm.mr.set_current_device_resource(mr)

    # Left
    shuffler = get_shuffler(left_shuffle_id)
    shuffler.wait_on(partition_id)
    left = pylibcudf_to_cudf_dataframe(
        unpack_and_concat(
            shuffler.extract(partition_id),
            stream=DEFAULT_STREAM,
            device_mr=mr,
        ),
        column_names=left_column_names,
    )
    # Right
    shuffler = get_shuffler(right_shuffle_id)
    shuffler.wait_on(partition_id)
    right = pylibcudf_to_cudf_dataframe(
        unpack_and_concat(
            shuffler.extract(partition_id),
            stream=DEFAULT_STREAM,
            device_mr=mr,
        ),
        column_names=right_column_names,
    )
    # Return merged result
    return left.merge(right, **merge_kwargs)


def shuffle(
    df: dd.DataFrame,
    on: Sequence[str],
    *,
    partition_count: int | None = None,
):
    """Shuffle data using a RAPIDS-MP shuffle service."""
    # Get client and shuffle id
    client = get_dask_client()
    shuffle_id = get_shuffle_id()
    meta = df._meta

    # Extract mapping between ranks and worker addresses
    worker_ranks: dict[int, str] = {
        v: k for k, v in client.run(get_worker_rank).items()
    }
    n_workers = len(worker_ranks)
    restricted_keys: MutableMapping[Any, str] = {}

    # Add operation to submit each partition to the shuffler
    partition_count = partition_count or df.optimize().npartitions
    df_id = df.map_partitions(
        rmp_shuffle_insert,
        on=on,
        partition_count=partition_count,
        shuffle_id=shuffle_id,
        meta=pd.DataFrame({"id": [shuffle_id]}),
        enforce_metadata=False,
    ).optimize()

    # Create task names
    token = tokenize(df_id, shuffle_id)
    global_barrier_name = f"rmp-global-barrier-{token}"
    worker_barrier_name = f"rmp-worker-barrier-{token}"
    extract_name = f"rmp-shuffle-extract-{token}"

    # Extract task graph and add global barrier task
    insert_keys = [(df_id._name, i) for i in range(df_id.npartitions)]
    dsk: MutableMapping[Any, Any] = {
        (global_barrier_name, 0): (
            global_rmp_barrier,
            (shuffle_id,),
            insert_keys,
        )
    }
    dsk.update(df_id.dask)

    # Add worker barrier tasks
    worker_barriers: MutableMapping[Any, Any] = {}
    for rank, addr in worker_ranks.items():
        key = (worker_barrier_name, rank)
        worker_barriers[rank] = key
        dsk[key] = (
            worker_rmp_barrier,
            (shuffle_id,),
            partition_count,
            (global_barrier_name, 0),
        )
        restricted_keys[key] = addr

    # Add extraction tasks
    output_keys = []
    column_names = list(meta.columns)
    for part_id in range(partition_count):
        rank = part_id % n_workers
        output_keys.append((extract_name, part_id))
        dsk[output_keys[-1]] = (
            rmp_shuffle_extract,
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

    # Construct/return a Dask-DataFrame collection
    divisions = (None,) * (partition_count + 1)
    name_prefix = "rmp-shuffle"
    return dd.from_graph(
        dsk,
        meta,
        divisions,
        output_keys,
        name_prefix,
    )


def merge(
    left: dd.DataFrame,
    right: dd.DataFrame,
    left_on: Sequence[str],
    right_on: Sequence[str],
    **kwargs,
):
    """Shuffle data using a RAPIDS-MP shuffle service."""
    # Get client and shuffle ids
    client = get_dask_client()
    left_shuffle_id = get_shuffle_id()
    right_shuffle_id = get_shuffle_id()
    left_meta = left._meta
    right_meta = right._meta
    meta = left._meta.merge(right._meta, left_on=left_on, right_on=right_on, **kwargs)

    # Extract mapping between ranks and worker addresses
    worker_ranks: dict[int, str] = {
        v: k for k, v in client.run(get_worker_rank).items()
    }
    n_workers = len(worker_ranks)
    restricted_keys: MutableMapping[Any, str] = {}

    # Add operation to submit each partition to the shuffler
    partition_count = max(left.optimize().npartitions, right.optimize().npartitions)
    df_id = dd.map_partitions(
        rmp_merge_insert,
        left,
        right,
        left_on,
        right_on,
        left_shuffle_id,
        right_shuffle_id,
        partition_count,
        meta=pd.DataFrame({"id": [0]}),
        enforce_metadata=False,
    ).optimize()

    # Create task names
    token = tokenize(df_id, left_shuffle_id, right_shuffle_id)
    global_barrier_name = f"rmp-global-barrier-{token}"
    worker_barrier_name = f"rmp-worker-barrier-{token}"
    extract_name = f"rmp-shuffle-extract-{token}"

    # Extract task graph and add global barrier task
    insert_keys = [(df_id._name, i) for i in range(df_id.npartitions)]
    dsk: MutableMapping[Any, Any] = {
        (global_barrier_name, 0): (
            global_rmp_barrier,
            (left_shuffle_id, right_shuffle_id),
            partition_count,
            insert_keys,
        )
    }
    dsk.update(df_id.dask)

    # Add worker barrier tasks
    worker_barriers: MutableMapping[Any, Any] = {}
    for rank, addr in worker_ranks.items():
        key = (worker_barrier_name, rank)
        worker_barriers[rank] = key
        dsk[key] = (
            worker_rmp_barrier,
            (left_shuffle_id, right_shuffle_id),
            partition_count,
            (global_barrier_name, 0),
        )
        restricted_keys[key] = addr

    # Add extraction tasks
    output_keys = []
    left_column_names = list(left_meta.columns)
    right_column_names = list(right_meta.columns)
    for part_id in range(partition_count):
        rank = part_id % n_workers
        output_keys.append((extract_name, part_id))
        dsk[output_keys[-1]] = (
            rmp_merge_extract,
            left_shuffle_id,
            right_shuffle_id,
            part_id,
            left_column_names,
            right_column_names,
            {"left_on": left_on, "right_on": right_on, **kwargs},
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

    # Construct/return a Dask-DataFrame collection
    divisions = (None,) * (partition_count + 1)
    name_prefix = "rmp-shuffle"
    return dd.from_graph(
        dsk,
        meta,
        divisions,
        output_keys,
        name_prefix,
    )
