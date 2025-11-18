# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import cudf
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmpf.integrations.cudf.partition import (
    partition_and_pack,
    spill_partitions,
    split_and_pack,
    unpack_and_concat,
    unspill_partitions,
)
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.testing import assert_eq
from rapidsmpf.utils.cudf import (
    cudf_to_pylibcudf_table,
    pylibcudf_to_cudf_dataframe,
)

if TYPE_CHECKING:
    import rmm.mr


@pytest.mark.parametrize("df", [{"0": [1, 2, 3], "1": [2, 2, 1]}, {"0": [], "1": []}])
@pytest.mark.parametrize("num_partitions", [1, 2, 3, 10])
def test_partition_and_pack_unpack(
    device_mr: rmm.mr.CudaMemoryResource, df: dict[str, list[int]], num_partitions: int
) -> None:
    br = BufferResource(device_mr)
    expect = cudf.DataFrame(df)
    partitions = partition_and_pack(
        cudf_to_pylibcudf_table(expect),
        columns_to_hash=(1,),
        num_partitions=num_partitions,
        br=br,
        stream=DEFAULT_STREAM,
    )
    got = pylibcudf_to_cudf_dataframe(
        unpack_and_concat(
            tuple(partitions.values()),
            br=br,
            stream=DEFAULT_STREAM,
        )
    )
    # Since the row order isn't preserved, we sort the rows by the "0" column.
    assert_eq(expect, got, sort_rows="0")


@pytest.mark.parametrize(
    "df",
    [
        {"0": [1, 2, 3], "1": [2, 2, 1]},
        {"0": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        {"0": [], "1": []},
    ],
)
@pytest.mark.parametrize("num_partitions", [1, 2, 3, 10])
def test_split_and_pack_unpack(
    device_mr: rmm.mr.CudaMemoryResource, df: dict[str, list[int]], num_partitions: int
) -> None:
    br = BufferResource(device_mr)
    expect = cudf.DataFrame(df)
    splits = np.linspace(0, len(expect), num_partitions, endpoint=False)[1:].astype(int)
    partitions = split_and_pack(
        cudf_to_pylibcudf_table(expect),
        splits=splits,
        br=br,
        stream=DEFAULT_STREAM,
    )
    got = pylibcudf_to_cudf_dataframe(
        unpack_and_concat(
            tuple(partitions[i] for i in range(num_partitions)),
            br=br,
            stream=DEFAULT_STREAM,
        )
    )

    assert_eq(expect, got)


@pytest.mark.parametrize("df", [{"0": [1, 2, 3], "1": [2, 2, 1]}, {"0": [], "1": []}])
@pytest.mark.parametrize("num_partitions", [1, 2, 3, 10])
def test_split_and_pack_unpack_out_of_range(
    device_mr: rmm.mr.CudaMemoryResource, df: dict[str, list[int]], num_partitions: int
) -> None:
    br = BufferResource(device_mr)
    expect = cudf.DataFrame({"0": [], "1": []})
    with pytest.raises(IndexError):
        split_and_pack(
            cudf_to_pylibcudf_table(expect),
            splits=[100],
            br=br,
            stream=DEFAULT_STREAM,
        )


@pytest.mark.parametrize("df", [{"0": [1, 2, 3], "1": [2, 2, 1]}, {"0": [], "1": []}])
@pytest.mark.parametrize("num_partitions", [1, 2, 3, 10])
def test_spill_unspill_roundtrip(
    device_mr: rmm.mr.CudaMemoryResource, df: dict[str, list[int]], num_partitions: int
) -> None:
    br = BufferResource(device_mr)
    expect = cudf.DataFrame(df)
    partitions = partition_and_pack(
        cudf_to_pylibcudf_table(expect),
        columns_to_hash=(1,),
        num_partitions=num_partitions,
        br=br,
        stream=DEFAULT_STREAM,
    )

    # Spill roundtrip
    spilled = spill_partitions(partitions.values(), br=br)
    unspilled = unspill_partitions(spilled, br=br, allow_overbooking=False)

    got = pylibcudf_to_cudf_dataframe(
        unpack_and_concat(
            unspilled,
            br=br,
            stream=DEFAULT_STREAM,
        )
    )
    # Since the row order isn't preserved, we sort the rows by the "0" column.
    assert_eq(expect, got, sort_rows="0")
