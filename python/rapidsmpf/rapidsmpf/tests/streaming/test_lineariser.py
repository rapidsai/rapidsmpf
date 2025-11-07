# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc

from rapidsmpf.streaming.core.leaf_node import pull_from_channel, push_to_channel
from rapidsmpf.streaming.core.lineariser import Lineariser
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import define_py_node, run_streaming_pipeline
from rapidsmpf.streaming.cudf.table_chunk import TableChunk
from rapidsmpf.testing import assert_eq

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.core.node import CppNode, PyNode


def test_lineariser(
    context: Context, stream: Stream, py_executor: ThreadPoolExecutor
) -> None:
    expects = [
        plc.Table([plc.Column.from_iterable_of_py([1 * seq, 2 * seq, 3 * seq])])
        for seq in range(10)
    ]
    table_chunks = [
        Message(
            seq, TableChunk.from_pylibcudf_table(expect, stream, exclusive_view=False)
        )
        for seq, expect in enumerate(expects)
    ]

    @define_py_node()
    async def drain(ctx: Context, lineariser: Lineariser) -> None:
        await lineariser.drain()

    ch_out: Channel[TableChunk] = context.create_channel()
    num_producers = 4
    lineariser = Lineariser(context, ch_out, 4)
    out, deferred = pull_from_channel(context, ch_in=ch_out)
    nodes: list[CppNode | PyNode] = [drain(context, lineariser), out]
    nodes.extend(
        push_to_channel(context, ch_out=ch_in, messages=table_chunks[i::num_producers])
        for i, ch_in in enumerate(lineariser.get_inputs())
    )

    run_streaming_pipeline(nodes=nodes, py_executor=py_executor)

    results = deferred.release()
    for seq, (result, expect) in enumerate(zip(results, expects, strict=True)):
        assert result.sequence_number == seq
        tbl = TableChunk.from_message(result)
        assert_eq(tbl.table_view(), expect)
