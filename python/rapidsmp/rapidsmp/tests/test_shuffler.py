# Copyright (c) 2025, NVIDIA CORPORATION.
from __future__ import annotations

import math

import numpy as np
import pytest
from mpi4py import MPI

import cudf
import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmp.buffer.resource import BufferResource
from rapidsmp.communicator.mpi import new_communicator
from rapidsmp.shuffler import Shuffler, partition_and_pack, unpack_and_concat
from rapidsmp.testing import assert_eq
from rapidsmp.utils.cudf import (
    cudf_to_pylibcudf_table,
    pylibcudf_to_cudf_dataframe,
)


@pytest.mark.parametrize("df", [{"0": [1, 2, 3], "1": [2, 2, 1]}, {"0": [], "1": []}])
@pytest.mark.parametrize("num_partitions", [1, 2, 3, 10])
def test_partition_and_pack_unpack(df, num_partitions):
    expect = cudf.DataFrame(df)
    partitions = partition_and_pack(
        cudf_to_pylibcudf_table(expect),
        columns_to_hash=(1,),
        num_partitions=num_partitions,
    )
    got = pylibcudf_to_cudf_dataframe(unpack_and_concat(tuple(partitions.values())))
    # Since the row order isn't preserved, we sort the rows by the "0" column.
    assert_eq(expect, got, sort_rows="0")


@pytest.mark.parametrize("total_num_partitions", [1, 2, 3, 10])
def test_shuffler_single_nonempty_partition(total_num_partitions):
    comm = new_communicator(MPI.COMM_WORLD)
    br = BufferResource(rmm.mr.get_current_device_resource())

    shuffler = Shuffler(
        comm, total_num_partitions=total_num_partitions, stream=DEFAULT_STREAM, br=br
    )

    df = cudf.DataFrame({"0": [1, 2, 3], "1": [42, 42, 42]})
    packed_inputs = partition_and_pack(
        cudf_to_pylibcudf_table(df),
        columns_to_hash=(df.columns.get_loc("1"),),
        num_partitions=total_num_partitions,
    )
    shuffler.insert_chunks(packed_inputs)

    for pid in range(total_num_partitions):
        shuffler.insert_finished(pid)

    local_outputs = []
    while not shuffler.finished():
        partition_id = shuffler.wait_any()
        packed_chunks = shuffler.extract(partition_id)
        partition = unpack_and_concat(packed_chunks)
        local_outputs.append(partition)
    shuffler.shutdown()
    # Everyting should go the a single rank thus we should get the whole dataframe or nothing.
    if len(local_outputs) == 0:
        return
    res = cudf.concat(
        [pylibcudf_to_cudf_dataframe(o) for o in local_outputs], ignore_index=True
    )
    # Each rank has `df` thus each rank contribute with the rows of `df` to the expected result.
    expect = cudf.concat([df] * MPI.COMM_WORLD.size, ignore_index=True)
    if not res.empty:
        assert_eq(res, expect, sort_rows="0")


@pytest.mark.parametrize("batch_size", [None, 10])
@pytest.mark.parametrize("total_num_partitions", [1, 2, 3, 10])
def test_shuffler_uniform(batch_size, total_num_partitions):
    mpi_comm = MPI.COMM_WORLD
    comm = new_communicator(mpi_comm)
    br = BufferResource(rmm.mr.get_current_device_resource())

    num_rows = 100
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
            unpack_and_concat([packed]),
            column_names=column_names,
        )
        for partition_id, packed in partition_and_pack(
            cudf_to_pylibcudf_table(df),
            columns_to_hash=columns_to_hash,
            num_partitions=total_num_partitions,
        ).items()
    }

    # Create shuffler
    shuffler = Shuffler(
        comm,
        total_num_partitions=total_num_partitions,
        stream=DEFAULT_STREAM,
        br=br,
    )

    # Slice df and submit local slices to shuffler
    stride = math.ceil(num_rows / mpi_comm.size)
    local_df = df.iloc[comm.rank * stride : (comm.rank + 1) * stride]
    num_rows_local = len(local_df)
    batch_size = batch_size or num_rows_local
    for i in range(0, num_rows_local, batch_size):
        packed_inputs = partition_and_pack(
            cudf_to_pylibcudf_table(local_df.iloc[i : i + batch_size]),
            columns_to_hash=columns_to_hash,
            num_partitions=total_num_partitions,
        )
        shuffler.insert_chunks(packed_inputs)

    # Tell shuffler we are done adding data
    for pid in range(total_num_partitions):
        shuffler.insert_finished(pid)

    # Extract and check shuffled partitions
    while not shuffler.finished():
        partition_id = shuffler.wait_any()
        packed_chunks = shuffler.extract(partition_id)
        partition = unpack_and_concat(packed_chunks)
        assert_eq(
            pylibcudf_to_cudf_dataframe(partition, column_names=column_names),
            expected[partition_id],
            sort_rows="a",
        )

    shuffler.shutdown()
