# Copyright (c) 2025, NVIDIA CORPORATION.
from __future__ import annotations

import pytest
from mpi4py import MPI

import cudf
import rmm.mr
from rmm._cuda.stream import DEFAULT_STREAM

from rapidsmp.buffer.resource import BufferResource
from rapidsmp.communicator.mpi import new_communicator
from rapidsmp.shuffler import Shuffler, partition_and_pack, unpack_and_concat
from rapidsmp.testing import assert_eq
from rapidsmp.utils.cudf import cudf_to_pylibcudf_table, pylibcudf_to_cudf_dataframe


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
        columns_to_hash=(1,),
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
    if not res.empty:
        assert_eq(res, df, sort_rows="0")
