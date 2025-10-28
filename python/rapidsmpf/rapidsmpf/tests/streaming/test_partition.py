# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import cudf

from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.leaf_node import pull_from_channel, push_to_channel
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import run_streaming_pipeline
from rapidsmpf.streaming.cudf.partition import partition_and_pack, unpack_and_concat
from rapidsmpf.streaming.cudf.table_chunk import TableChunk
from rapidsmpf.testing import assert_eq
from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.cudf.partition_chunk import PartitionMapChunk


@pytest.mark.parametrize("num_partitions", [1, 2, 3, 10])
def test_partition_and_pack_unpack(
    context: Context, stream: Stream, num_partitions: int
) -> None:
    expects = [
        cudf_to_pylibcudf_table(cudf.DataFrame({"0": [1, 2, 3], "1": [2, 2, 1]})),
        cudf_to_pylibcudf_table(cudf.DataFrame({"0": [], "1": []})),
    ]
    table_chunks = [
        Message(
            seq, TableChunk.from_pylibcudf_table(expect, stream, exclusive_view=False)
        )
        for seq, expect in enumerate(expects)
    ]
    ch1: Channel[TableChunk] = Channel()
    node1 = push_to_channel(context, ch_out=ch1, messages=table_chunks)

    ch2: Channel[PartitionMapChunk] = Channel()
    node2 = partition_and_pack(
        context,
        ch_in=ch1,
        ch_out=ch2,
        columns_to_hash=(1,),
        num_partitions=num_partitions,
    )

    ch3: Channel[TableChunk] = Channel()
    node3 = unpack_and_concat(
        context,
        ch_in=ch2,
        ch_out=ch3,
    )

    node4, output = pull_from_channel(context, ch_in=ch3)
    run_streaming_pipeline(nodes=(node1, node2, node3, node4))

    results = output.release()
    for seq, (result, expect) in enumerate(zip(results, expects, strict=True)):
        assert result.sequence_number == seq
        tbl = TableChunk.from_message(result)
        assert_eq(tbl.table_view(), expect, sort_rows="0")
