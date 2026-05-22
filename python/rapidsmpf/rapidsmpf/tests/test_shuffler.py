# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

import pylibcudf as plc
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmpf.integrations.cudf.partition import (
    partition_and_pack,
    unpack_and_concat,
    unspill_partitions,
)
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.shuffler import (
    Shuffler,
)
from rapidsmpf.testing import assert_eq_with_plc

if TYPE_CHECKING:
    import rmm.mr

    from rapidsmpf.communicator.communicator import Communicator


@pytest.mark.parametrize("total_num_partitions", [1, 2, 3, 10])
def test_shuffler_single_nonempty_partition(
    comm: Communicator,
    device_mr: rmm.mr.CudaMemoryResource,
    total_num_partitions: int,
) -> None:
    br = BufferResource(device_mr)

    shuffler = Shuffler(
        comm,
        op_id=0,
        total_num_partitions=total_num_partitions,
        br=br,
    )

    df = plc.Table(
        [
            plc.Column.from_iterable_of_py([1, 2, 3], plc.DataType(plc.TypeId.INT64)),
            plc.Column.from_iterable_of_py(
                [42, 42, 42], plc.DataType(plc.TypeId.INT64)
            ),
        ]
    )
    packed_inputs = partition_and_pack(
        df,
        columns_to_hash=(1,),
        num_partitions=total_num_partitions,
        br=br,
        stream=DEFAULT_STREAM,
    )
    shuffler.insert_chunks(packed_inputs)
    shuffler.insert_finished()

    expected_partitions = set(shuffler.local_partitions())

    local_outputs = []
    extracted_partitions = set()
    shuffler.wait()
    for partition_id in shuffler.local_partitions():
        extracted_partitions.add(partition_id)
        packed_chunks = shuffler.extract(partition_id)
        partition = unpack_and_concat(
            unspill_partitions(packed_chunks, br=br, allow_overbooking=True),
            br=br,
            stream=DEFAULT_STREAM,
        )
        local_outputs.append(partition)
    shuffler.shutdown()
    assert extracted_partitions == expected_partitions
    # Everything should go to a single rank thus we should get the whole dataframe or nothing.
    if len(local_outputs) == 0:
        return
    res = plc.concatenate.concatenate(local_outputs)
    # Each rank has `df` thus each rank contribute to the rows of `df` to the expected result.
    expect = plc.concatenate.concatenate([df] * comm.nranks)
    if res.num_rows() > 0:
        assert_eq_with_plc(res, expect, sort_rows=0)


@pytest.mark.parametrize("batch_size", [None, 10])
@pytest.mark.parametrize("total_num_partitions", [1, 2, 3, 10])
def test_shuffler_uniform(
    comm: Communicator,
    device_mr: rmm.mr.CudaMemoryResource,
    batch_size: int | None,
    total_num_partitions: int,
) -> None:
    br = BufferResource(device_mr)

    # Every rank creates the full input dataframe and all the expected partitions
    # (also partitions this rank might not get after the shuffle).
    num_rows = 100
    np.random.seed(42)  # Make sure all ranks create the same input dataframe.
    df = plc.Table(
        [
            plc.Column.from_iterable_of_py(
                list(range(num_rows)), plc.DataType(plc.TypeId.INT64)
            ),
            plc.Column.from_array(np.random.randint(0, 1000, num_rows)),
            plc.Column.from_iterable_of_py(
                ["cat", "dog"] * (num_rows // 2), plc.DataType(plc.TypeId.STRING)
            ),
        ]
    )
    columns_to_hash = (1,)

    expected = {
        partition_id: unpack_and_concat(
            [packed],
            br=br,
            stream=DEFAULT_STREAM,
        )
        for partition_id, packed in partition_and_pack(
            df,
            columns_to_hash=columns_to_hash,
            num_partitions=total_num_partitions,
            br=br,
            stream=DEFAULT_STREAM,
        ).items()
    }

    shuffler = Shuffler(
        comm,
        op_id=0,
        total_num_partitions=total_num_partitions,
        br=br,
    )

    # Slice df and submit local slices to shuffler
    stride = math.ceil(num_rows / comm.nranks)
    local_df = plc.copying.slice(df, [comm.rank * stride, (comm.rank + 1) * stride])[0]
    num_rows_local = local_df.num_rows()
    batch_size = batch_size or num_rows_local
    for i in range(0, num_rows_local, batch_size):
        batch = plc.copying.slice(local_df, [i, i + batch_size])[0]
        packed_inputs = partition_and_pack(
            batch,
            columns_to_hash=columns_to_hash,
            num_partitions=total_num_partitions,
            br=br,
            stream=DEFAULT_STREAM,
        )
        shuffler.insert_chunks(packed_inputs)

    # Tell shuffler we are done adding data
    shuffler.insert_finished()

    expected_partitions = set(shuffler.local_partitions())
    extracted_partitions = set()
    shuffler.wait()
    for partition_id in shuffler.local_partitions():
        extracted_partitions.add(partition_id)
        packed_chunks = shuffler.extract(partition_id)
        partition = unpack_and_concat(
            unspill_partitions(packed_chunks, br=br, allow_overbooking=True),
            br=br,
            stream=DEFAULT_STREAM,
        )
        assert_eq_with_plc(
            partition,
            expected[partition_id],
            sort_rows=0,
        )

    shuffler.shutdown()
    assert extracted_partitions == expected_partitions
