# Copyright (c) 2025, NVIDIA CORPORATION.

import cudf

from rapidsmp.shuffler import partition_and_pack, unpack_and_concat
from rapidsmp.testing import assert_eq, to_cudf_dataframe, to_pylibcudf_table


def test_partition_and_pack_unpack():
    expect = cudf.DataFrame({"0": [1, 2, 3], "1": [2, 2, 1]})

    partitions = partition_and_pack(
        to_pylibcudf_table(expect), columns_to_hash=(1,), num_partitions=2
    )
    got = to_cudf_dataframe(unpack_and_concat(tuple(partitions.values())))
    assert_eq(expect.sort_values(by="0"), got.sort_values(by="0"))
