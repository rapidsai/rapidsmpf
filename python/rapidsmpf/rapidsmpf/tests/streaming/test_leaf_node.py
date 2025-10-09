# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import cudf

from rapidsmpf.streaming.core.channel import Channel, Message
from rapidsmpf.streaming.core.leaf_node import pull_from_channel, push_to_channel
from rapidsmpf.streaming.core.node import run_streaming_pipeline
from rapidsmpf.streaming.cudf.table_chunk import TableChunk
from rapidsmpf.testing import assert_eq
from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.streaming.core.context import Context


def test_roundtrip(context: Context, stream: Stream) -> None:
    expects = [
        cudf_to_pylibcudf_table(cudf.DataFrame({"a": [1 * seq, 2 * seq, 3 * seq]}))
        for seq in range(10)
    ]
    table_chunks = [
        Message(
            TableChunk.from_pylibcudf_table(seq, expect, stream, exclusive_view=False)
        )
        for seq, expect in enumerate(expects)
    ]
    ch1: Channel[TableChunk] = Channel()
    node1 = push_to_channel(context, ch_out=ch1, messages=table_chunks)
    node2, output = pull_from_channel(context, ch_in=ch1)
    run_streaming_pipeline(nodes=(node1, node2))

    results = output.release()
    for seq, (result, expect) in enumerate(zip(results, expects, strict=True)):
        tbl = TableChunk.from_message(result)
        assert tbl.sequence_number == seq
        assert_eq(tbl.table_view(), expect)
