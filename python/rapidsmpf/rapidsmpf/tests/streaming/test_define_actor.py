# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import cudf

from rapidsmpf.streaming.core.actor import define_actor, run_actor_graph
from rapidsmpf.streaming.core.leaf_actor import pull_from_channel, push_to_channel
from rapidsmpf.streaming.core.message import Message
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

    # The actor access `ch1` both through the `ch_out` parameter and the closure.
    @define_actor(extra_channels=(ch1,))
    async def actor1(ctx: Context, /, ch_out: Channel) -> None:
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

    actor2, output = pull_from_channel(context, ch_in=ch1)

    run_actor_graph(
        actors=[
            actor1(context, ch_out=ch1),
            actor2,
        ],
        py_executor=py_executor,
    )

    results = output.release()
    for seq, (result, expect) in enumerate(zip(results, expects, strict=True)):
        assert result.sequence_number == seq
        tbl = TableChunk.from_message(result)
        assert_eq(tbl.table_view(), expect)


def test_shutdown(context: Context, py_executor: ThreadPoolExecutor) -> None:
    @define_actor()
    async def actor1(ctx: Context, ch_out: Channel[TableChunk]) -> None:
        await ch_out.shutdown(ctx)
        # Calling shutdown multiple times is allowed.
        await ch_out.shutdown(ctx)

    ch1: Channel[TableChunk] = context.create_channel()
    actor2, output = pull_from_channel(context, ch_in=ch1)

    run_actor_graph(
        actors=[
            actor1(context, ch_out=ch1),
            actor2,
        ],
        py_executor=py_executor,
    )

    assert output.release() == []


def test_send_error(context: Context, py_executor: ThreadPoolExecutor) -> None:
    @define_actor()
    async def actor1(ctx: Context, ch_out: Channel[TableChunk]) -> None:
        raise RuntimeError("MyError")

    ch1: Channel[TableChunk] = context.create_channel()
    actor2, output = pull_from_channel(context, ch_in=ch1)

    with pytest.raises(
        RuntimeError,
        match="MyError",
    ):
        run_actor_graph(
            actors=[
                actor1(context, ch_out=ch1),
                actor2,
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

    @define_actor()
    async def actor1(ctx: Context, ch_in: Channel[TableChunk]) -> None:
        while True:
            chunk = await ch_in.recv(context)
            if chunk is None:
                break
            results.append(chunk)

    ch1: Channel[TableChunk] = context.create_channel()

    run_actor_graph(
        actors=[
            push_to_channel(context, ch_out=ch1, messages=table_chunks),
            actor1(context, ch_in=ch1),
        ],
        py_executor=py_executor,
    )

    for seq, (result, expect) in enumerate(zip(results, expects, strict=True)):
        assert result.sequence_number == seq
        tbl = TableChunk.from_message(result)
        assert_eq(tbl.table_view(), expect)
