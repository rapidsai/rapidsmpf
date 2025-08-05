# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

import cudf
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmpf.buffer.resource import BufferResource
from rapidsmpf.integrations.cudf.partition import (
    partition_and_pack,
    unpack_and_concat,
    unspill_partitions,
)
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.shuffler import (
    Shuffler,
)
from rapidsmpf.testing import assert_eq
from rapidsmpf.utils.cudf import (
    cudf_to_pylibcudf_table,
    pylibcudf_to_cudf_dataframe,
)

if TYPE_CHECKING:
    import rmm.mr

    from rapidsmpf.communicator.communicator import Communicator


@pytest.mark.parametrize("wait_on", [False, True])
@pytest.mark.parametrize("total_num_partitions", [1, 2, 3, 10])
@pytest.mark.parametrize("concat", [False, True])
def test_shuffler_single_nonempty_partition(
    comm: Communicator,
    device_mr: rmm.mr.CudaMemoryResource,
    total_num_partitions: int,
    wait_on: bool,  # noqa: FBT001
    concat: bool,  # noqa: FBT001
) -> None:
    br = BufferResource(device_mr)
    progress_thread = ProgressThread(comm)

    shuffler = Shuffler(
        comm,
        progress_thread,
        op_id=0,
        total_num_partitions=total_num_partitions,
        stream=DEFAULT_STREAM,
        br=br,
    )

    df = cudf.DataFrame({"0": [1, 2, 3], "1": [42, 42, 42]})
    packed_inputs = partition_and_pack(
        cudf_to_pylibcudf_table(df),
        columns_to_hash=(df.columns.get_loc("1"),),
        num_partitions=total_num_partitions,
        br=br,
        stream=DEFAULT_STREAM,
    )
    if concat:
        shuffler.concat_insert(packed_inputs)
    else:
        shuffler.insert_chunks(packed_inputs)

    my_partitions = {
        p for p in range(total_num_partitions) if (p % comm.nranks) == comm.rank
    }
    if concat:
        shuffler.insert_finished(list(range(total_num_partitions)))
    else:
        for pid in range(total_num_partitions):
            shuffler.insert_finished(pid)

    local_outputs = []
    while not shuffler.finished():
        if wait_on:
            # Wait on a specific partition id
            partition_id = my_partitions.pop()
            shuffler.wait_on(partition_id)
        else:
            # Wait on any partition id
            partition_id = shuffler.wait_any()
            my_partitions.remove(partition_id)
        packed_chunks = shuffler.extract(partition_id)
        partition = unpack_and_concat(
            unspill_partitions(
                packed_chunks, stream=DEFAULT_STREAM, br=br, allow_overbooking=True
            ),
            br=br,
            stream=DEFAULT_STREAM,
        )
        local_outputs.append(partition)
    shuffler.shutdown()
    # Everyting should go the a single rank thus we should get the whole dataframe or nothing.
    if len(local_outputs) == 0:
        return
    res = cudf.concat(
        [pylibcudf_to_cudf_dataframe(o) for o in local_outputs], ignore_index=True
    )
    # Each rank has `df` thus each rank contribute to the rows of `df` to the expected result.
    expect = cudf.concat([df] * comm.nranks, ignore_index=True)
    if not res.empty:
        assert_eq(res, expect, sort_rows="0")


@pytest.mark.parametrize("batch_size", [None, 10])
@pytest.mark.parametrize("total_num_partitions", [1, 2, 3, 10])
@pytest.mark.parametrize("concat", [False, True])
def test_shuffler_uniform(
    comm: Communicator,
    device_mr: rmm.mr.CudaMemoryResource,
    batch_size: int | None,
    total_num_partitions: int,
    concat: bool,  # noqa: FBT001
) -> None:
    br = BufferResource(device_mr)

    # Every rank creates the full input dataframe and all the expected partitions
    # (also partitions this rank might not get after the shuffle).
    num_rows = 100
    np.random.seed(42)  # Make sure all ranks create the same input dataframe.
    df = cudf.DataFrame(
        {
            "a": range(num_rows),
            "b": np.random.randint(0, 1000, num_rows),
            "c": ["cat", "dog"] * (num_rows // 2),
        }
    )
    columns_to_hash = (df.columns.get_loc("b"),)
    column_names = list(df.columns)

    # Calculate the expected output partitions on all ranks
    expected = {
        partition_id: pylibcudf_to_cudf_dataframe(
            unpack_and_concat(
                [packed],
                br=br,
                stream=DEFAULT_STREAM,
            ),
            column_names=column_names,
        )
        for partition_id, packed in partition_and_pack(
            cudf_to_pylibcudf_table(df),
            columns_to_hash=columns_to_hash,
            num_partitions=total_num_partitions,
            br=br,
            stream=DEFAULT_STREAM,
        ).items()
    }

    progress_thread = ProgressThread(comm)

    # Create shuffler
    shuffler = Shuffler(
        comm,
        progress_thread,
        op_id=0,
        total_num_partitions=total_num_partitions,
        stream=DEFAULT_STREAM,
        br=br,
    )

    # Slice df and submit local slices to shuffler
    stride = math.ceil(num_rows / comm.nranks)
    local_df = df.iloc[comm.rank * stride : (comm.rank + 1) * stride]
    num_rows_local = len(local_df)
    batch_size = batch_size or num_rows_local
    for i in range(0, num_rows_local, batch_size):
        packed_inputs = partition_and_pack(
            cudf_to_pylibcudf_table(local_df.iloc[i : i + batch_size]),
            columns_to_hash=columns_to_hash,
            num_partitions=total_num_partitions,
            br=br,
            stream=DEFAULT_STREAM,
        )
        if concat:
            shuffler.concat_insert(packed_inputs)
        else:
            shuffler.insert_chunks(packed_inputs)

    # Tell shuffler we are done adding data
    if concat:
        shuffler.insert_finished(list(range(total_num_partitions)))
    else:
        for pid in range(total_num_partitions):
            shuffler.insert_finished(pid)

    # Extract and check shuffled partitions
    while not shuffler.finished():
        partition_id = shuffler.wait_any()
        packed_chunks = shuffler.extract(partition_id)
        partition = unpack_and_concat(
            unspill_partitions(
                packed_chunks, stream=DEFAULT_STREAM, br=br, allow_overbooking=True
            ),
            br=br,
            stream=DEFAULT_STREAM,
        )
        assert_eq(
            pylibcudf_to_cudf_dataframe(partition, column_names=column_names),
            expected[partition_id],
            sort_rows="a",
        )

    shuffler.shutdown()
