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
from rapidsmp.testing import assert_eq, to_cudf_dataframe, to_pylibcudf_table


@pytest.mark.parametrize("df", [{"0": [1, 2, 3], "1": [2, 2, 1]}, {"0": [], "1": []}])
@pytest.mark.parametrize("num_partitions", [1, 2, 3, 10])
def test_partition_and_pack_unpack(df, num_partitions):
    expect = cudf.DataFrame(df)
    partitions = partition_and_pack(
        to_pylibcudf_table(expect), columns_to_hash=(1,), num_partitions=num_partitions
    )
    got = to_cudf_dataframe(unpack_and_concat(tuple(partitions.values())))
    # Since the row order isn't preserved, we sort before comparing.
    assert_eq(expect.sort_values(by="0"), got.sort_values(by="0"))


def test_shuffler():
    total_num_partitions = 2
    comm = new_communicator(MPI.COMM_WORLD)
    br = BufferResource(rmm.mr.get_current_device_resource())

    shuffler = Shuffler(
        comm, total_num_partitions=total_num_partitions, stream=DEFAULT_STREAM, br=br
    )

    df = cudf.DataFrame({"0": [1, 2, 3], "1": [1, 2, 2]})
    packed_inputs = partition_and_pack(
        to_pylibcudf_table(df),
        columns_to_hash=(1,),
        num_partitions=total_num_partitions,
    )
    shuffler.insert_chunks(packed_inputs)

    for pid in range(total_num_partitions):
        shuffler.insert_finished(pid)

    output_partitions = []
    while not shuffler.finished():
        partition_id = shuffler.wait_any()
        packed_chunks = shuffler.extract(partition_id)
        partition = unpack_and_concat(packed_chunks)
        output_partitions.append(partition)
    shuffler.shutdown()
