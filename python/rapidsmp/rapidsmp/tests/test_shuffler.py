# Copyright (c) 2025, NVIDIA CORPORATION.
from __future__ import annotations

from mpi4py import MPI

import cudf
import rmm.mr
from rmm._cuda.stream import DEFAULT_STREAM

from rapidsmp.buffer.resource import BufferResource
from rapidsmp.communicator.mpi import new_communicator
from rapidsmp.shuffler import Shuffler, partition_and_pack, unpack_and_concat
from rapidsmp.testing import assert_eq, to_cudf_dataframe, to_pylibcudf_table


def test_partition_and_pack_unpack():
    expect = cudf.DataFrame({"0": [1, 2, 3], "1": [2, 2, 1]})

    partitions = partition_and_pack(
        to_pylibcudf_table(expect), columns_to_hash=(1,), num_partitions=2
    )
    got = to_cudf_dataframe(unpack_and_concat(tuple(partitions.values())))
    assert_eq(expect.sort_values(by="0"), got.sort_values(by="0"))


def test_shuffler():
    total_num_partitions = 2
    comm = new_communicator(MPI.COMM_WORLD)
    br = BufferResource(rmm.mr.get_current_device_resource())

    shuffler = Shuffler(
        comm, total_num_partitions=total_num_partitions, stream=DEFAULT_STREAM, br=br
    )
    print(shuffler)

    df = cudf.DataFrame({"0": [1, 2, 3], "1": [1, 2, 2]})
    packed_inputs = partition_and_pack(
        to_pylibcudf_table(df),
        columns_to_hash=(1,),
        num_partitions=total_num_partitions,
    )
    shuffler.insert_chunks(packed_inputs)

    for pid in range(total_num_partitions):
        shuffler.insert_finished(pid)
