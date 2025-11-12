# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

import cudf

from rapidsmpf.streaming.chunks.arbitrary import ArbitraryChunk
from rapidsmpf.streaming.core.leaf_node import pull_from_channel, push_to_channel
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import define_py_node, run_streaming_pipeline
from rapidsmpf.streaming.cudf.table_chunk import TableChunk
from rapidsmpf.testing import assert_eq
from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context


def test_send_table_chunks(
    context: Context, stream: Stream, py_executor: ThreadPoolExecutor
) -> None:
    expects = [
        cudf_to_pylibcudf_table(cudf.DataFrame({"a": [1 * seq, 2 * seq, 3 * seq]}))
        for seq in range(10)
    ]

    ch1: Channel[TableChunk] = context.create_channel()

    # The node access `ch1` both through the `ch_out` parameter and the closure.
    @define_py_node(extra_channels=(ch1,))
    async def node1(ctx: Context, /, ch_out: Channel) -> None:
        for seq, chunk in enumerate(expects):
            await ch1.send(
                context,
                Message(
                    seq,
                    TableChunk.from_pylibcudf_table(
                        table=chunk,
                        stream=stream,
                        exclusive_view=False,
                    ),
                ),
            )
        await ch_out.drain(context)

    node2, output = pull_from_channel(context, ch_in=ch1)

    run_streaming_pipeline(
        nodes=[
            node1(context, ch_out=ch1),
            node2,
        ],
        py_executor=py_executor,
    )

    results = output.release()
    for seq, (result, expect) in enumerate(zip(results, expects, strict=True)):
        assert result.sequence_number == seq
        tbl = TableChunk.from_message(result)
        assert_eq(tbl.table_view(), expect)


def test_shutdown(context: Context, py_executor: ThreadPoolExecutor) -> None:
    @define_py_node()
    async def node1(ctx: Context, ch_out: Channel[TableChunk]) -> None:
        await ch_out.shutdown(ctx)
        # Calling shutdown multiple times is allowed.
        await ch_out.shutdown(ctx)

    ch1: Channel[TableChunk] = context.create_channel()
    node2, output = pull_from_channel(context, ch_in=ch1)

    run_streaming_pipeline(
        nodes=[
            node1(context, ch_out=ch1),
            node2,
        ],
        py_executor=py_executor,
    )

    assert output.release() == []


def test_send_error(context: Context, py_executor: ThreadPoolExecutor) -> None:
    @define_py_node()
    async def node1(ctx: Context, ch_out: Channel[TableChunk]) -> None:
        raise RuntimeError("MyError")

    ch1: Channel[TableChunk] = context.create_channel()
    node2, output = pull_from_channel(context, ch_in=ch1)

    with pytest.raises(
        RuntimeError,
        match="MyError",
    ):
        run_streaming_pipeline(
            nodes=[
                node1(context, ch_out=ch1),
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
        Message(
            seq, TableChunk.from_pylibcudf_table(expect, stream, exclusive_view=False)
        )
        for seq, expect in enumerate(expects)
    ]

    results: list[Message[TableChunk]] = []

    @define_py_node()
    async def node1(ctx: Context, ch_in: Channel[TableChunk]) -> None:
        while True:
            chunk = await ch_in.recv(context)
            if chunk is None:
                break
            results.append(chunk)

    ch1: Channel[TableChunk] = context.create_channel()

    run_streaming_pipeline(
        nodes=[
            push_to_channel(context, ch_out=ch1, messages=table_chunks),
            node1(context, ch_in=ch1),
        ],
        py_executor=py_executor,
    )

    for seq, (result, expect) in enumerate(zip(results, expects, strict=True)):
        assert result.sequence_number == seq
        tbl = TableChunk.from_message(result)
        assert_eq(tbl.table_view(), expect)


def test_pynode_raises_exception(
    context: Context, py_executor: ThreadPoolExecutor
) -> None:
    # https://github.com/rapidsai/rapidsmpf/issues/655
    # Previously, this would regularly segfault.

    if sys.version_info >= (3, 11):
        exc = ExceptionGroup  # noqa: F821
    else:
        exc = RuntimeError

    @define_py_node()
    async def sender(ctx: Context, ch_in: Channel, ch_out: Channel) -> None:
        # send some bytes to ch_out
        for i in range(4):
            await ch_out.send(ctx, Message(i, ArbitraryChunk(f"hello {i}".encode())))

        raise RuntimeError("sender error")

    @define_py_node()
    async def receiver(ctx: Context, ch_in: Channel) -> None:
        while True:
            msg = await ch_in.recv(ctx)
            if msg is None:
                break
            ArbitraryChunk.from_message(msg).release()

    ch_in: Channel[ArbitraryChunk[bytes]] = context.create_channel()
    ch_out: Channel[ArbitraryChunk[bytes]] = context.create_channel()

    with pytest.raises(
        exc, match="Exceptions in rapidsmpf.streaming.node.run_streaming_pipeline"
    ):
        run_streaming_pipeline(
            nodes=[
                sender(context, ch_in, ch_out),
                receiver(context, ch_out),
            ],
            py_executor=py_executor,
        )
