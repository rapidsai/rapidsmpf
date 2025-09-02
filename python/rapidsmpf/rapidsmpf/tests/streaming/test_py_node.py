# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import cudf

from rapidsmpf.streaming.core.channel import Channel, Message
from rapidsmpf.streaming.core.leaf_node import pull_from_channel, push_to_channel
from rapidsmpf.streaming.core.node import define_py_node, run_streaming_pipeline
from rapidsmpf.streaming.cudf.table_chunk import TableChunk
from rapidsmpf.testing import assert_eq
from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.streaming.core.context import Context


def test_send_table_chunks(
    context: Context, stream: Stream, py_executor: ThreadPoolExecutor
) -> None:
    expects = [
        cudf_to_pylibcudf_table(cudf.DataFrame({"a": [1 * seq, 2 * seq, 3 * seq]}))
        for seq in range(10)
    ]

    ch1: Channel[TableChunk] = Channel()

    # The node access `ch1` both from coroutine a parameter and the closure.
    @define_py_node(context, channels=(ch1,))
    async def node1(ch_out: Channel) -> None:
        for seq, chunk in enumerate(expects):
            await ch1.send(
                context,
                Message(
                    TableChunk.from_pylibcudf_table(
                        sequence_number=seq,
                        table=chunk,
                        stream=stream,
                    )
                ),
            )
        await ch_out.drain(context)

    node2, output = pull_from_channel(ctx=context, ch_in=ch1)

    run_streaming_pipeline(
        nodes=[
            node1(ch_out=ch1),
            node2,
        ],
        py_executor=py_executor,
    )

    results = output.release()
    for seq, (result, expect) in enumerate(zip(results, expects, strict=True)):
        tbl = TableChunk.from_message(result)
        assert tbl.sequence_number() == seq
        assert_eq(tbl.table_view(), expect)


def test_shutdown(context: Context, py_executor: ThreadPoolExecutor) -> None:
    @define_py_node(context)
    async def node1(ch_out: Channel[TableChunk]) -> None:
        await ch_out.shutdown(context)
        # Calling shutdown multiple times is allowed.
        await ch_out.shutdown(context)

    ch1: Channel[TableChunk] = Channel()
    node2, output = pull_from_channel(ctx=context, ch_in=ch1)

    run_streaming_pipeline(
        nodes=[
            node1(ch_out=ch1),
            node2,
        ],
        py_executor=py_executor,
    )

    assert output.release() == []


def test_send_error(context: Context, py_executor: ThreadPoolExecutor) -> None:
    @define_py_node(context)
    async def node1(ch_out: Channel[TableChunk]) -> None:
        raise RuntimeError("MyError")

    ch1: Channel[TableChunk] = Channel()
    node2, output = pull_from_channel(ctx=context, ch_in=ch1)

    with pytest.raises(
        RuntimeError,
        match="MyError",
    ):
        run_streaming_pipeline(
            nodes=[
                node1(ch_out=ch1),
                node2,
            ],
            py_executor=py_executor,
        )

    assert output.release() == []


def test_recv_table_chunks(
    context: Context, stream: Stream, py_executor: ThreadPoolExecutor
) -> None:
    expects = [
        cudf_to_pylibcudf_table(cudf.DataFrame({"a": [1 * seq, 2 * seq, 3 * seq]}))
        for seq in range(10)
    ]
    table_chunks = [
        Message(TableChunk.from_pylibcudf_table(seq, expect, stream))
        for seq, expect in enumerate(expects)
    ]

    results: list[Message[TableChunk]] = []

    @define_py_node(context)
    async def node1(ch_in: Channel[TableChunk]) -> None:
        while True:
            chunk = await ch_in.recv(context)
            if chunk is None:
                break
            results.append(chunk)

    ch1: Channel[TableChunk] = Channel()

    run_streaming_pipeline(
        nodes=[
            push_to_channel(ctx=context, ch_out=ch1, messages=table_chunks),
            node1(ch_in=ch1),
        ],
        py_executor=py_executor,
    )

    for seq, (result, expect) in enumerate(zip(results, expects, strict=True)):
        tbl = TableChunk.from_message(result)
        assert tbl.sequence_number() == seq
        assert_eq(tbl.table_view(), expect)
