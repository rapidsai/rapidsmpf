# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rapidsmpf.streaming.chunks.arbitrary import ArbitraryChunk
from rapidsmpf.streaming.core.actor import define_actor, run_actor_graph
from rapidsmpf.streaming.core.message import Message

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    from rapidsmpf.streaming.core.actor import PyActor
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context


def make_messages(start: int, count: int) -> list[Message[ArbitraryChunk[int]]]:
    return [
        Message(start + offset, ArbitraryChunk(start + offset))
        for offset in range(count)
    ]


@define_actor()
async def send_data(
    ctx: Context,
    ch_out: Channel[ArbitraryChunk[int]],
    start: int,
    count: int,
) -> None:
    for msg in make_messages(start, count):
        await ch_out.send(ctx, msg)
    await ch_out.drain(ctx)


@define_actor()
async def send_metadata_then_data(
    ctx: Context,
    ch_out: Channel[ArbitraryChunk[int]],
    metadata_values: list[int],
    data_start: int,
    data_count: int,
) -> None:
    for seq, value in enumerate(metadata_values):
        await ch_out.send_metadata(ctx, Message(seq, ArbitraryChunk(value)))
    await ch_out.shutdown_metadata(ctx)
    for msg in make_messages(data_start, data_count):
        await ch_out.send(ctx, msg)
    await ch_out.drain(ctx)


@define_actor()
async def send_data_only_with_metadata_shutdown(
    ctx: Context,
    ch_out: Channel[ArbitraryChunk[int]],
    data_start: int,
    data_count: int,
) -> None:
    await ch_out.shutdown_metadata(ctx)
    for msg in make_messages(data_start, data_count):
        await ch_out.send(ctx, msg)
    await ch_out.drain(ctx)


@define_actor()
async def send_metadata_only(
    ctx: Context, ch_out: Channel[ArbitraryChunk[int]], metadata_values: list[int]
) -> None:
    for seq, value in enumerate(metadata_values):
        await ch_out.send_metadata(ctx, Message(seq, ArbitraryChunk(value)))
    await ch_out.drain(ctx)


@define_actor()
async def consume_metadata_then_data(
    ctx: Context,
    ch_in: Channel[ArbitraryChunk[int]],
    metadata_out: list[int],
    data_out: list[int],
) -> None:
    while (msg := await ch_in.recv_metadata(ctx)) is not None:
        metadata_out.append(ArbitraryChunk.from_message(msg).release())

    while (msg := await ch_in.recv(ctx)) is not None:
        data_out.append(ArbitraryChunk.from_message(msg).release())


def test_data_roundtrip_without_metadata(
    context: Context, py_executor: ThreadPoolExecutor
) -> None:
    ch: Channel[ArbitraryChunk[int]] = context.create_channel()
    outputs: list[int] = []
    nodes: list[PyActor] = [
        send_data(context, ch, 0, 3),
    ]

    @define_actor()
    async def consume_only_data(
        ctx: Context, ch_in: Channel[ArbitraryChunk[int]]
    ) -> None:
        while (msg := await ch_in.recv(ctx)) is not None:
            outputs.append(ArbitraryChunk.from_message(msg).release())

    nodes.append(consume_only_data(context, ch))
    run_actor_graph(nodes=nodes, py_executor=py_executor)

    assert outputs == [0, 1, 2]


def test_metadata_then_data_roundtrip(
    context: Context, py_executor: ThreadPoolExecutor
) -> None:
    ch: Channel[ArbitraryChunk[int]] = context.create_channel()
    metadata_out: list[int] = []
    data_out: list[int] = []

    nodes: list[PyActor] = [
        send_metadata_then_data(context, ch, [10, 20], 0, 2),
        consume_metadata_then_data(context, ch, metadata_out, data_out),
    ]
    run_actor_graph(nodes=nodes, py_executor=py_executor)

    assert metadata_out == [10, 20]
    assert data_out == [0, 1]


def test_data_only_with_metadata_shutdown(
    context: Context, py_executor: ThreadPoolExecutor
) -> None:
    ch: Channel[ArbitraryChunk[int]] = context.create_channel()
    metadata_out: list[int] = []
    data_out: list[int] = []
    nodes: list[PyActor] = [
        send_data_only_with_metadata_shutdown(context, ch, 5, 2),
        consume_metadata_then_data(context, ch, metadata_out, data_out),
    ]
    run_actor_graph(nodes=nodes, py_executor=py_executor)

    assert metadata_out == []
    assert data_out == [5, 6]


def test_metadata_only_with_data_shutdown(
    context: Context, py_executor: ThreadPoolExecutor
) -> None:
    ch: Channel[ArbitraryChunk[int]] = context.create_channel()
    metadata_out: list[int] = []
    data_out: list[int] = []
    nodes: list[PyActor] = [
        send_metadata_only(context, ch, [30, 31]),
        consume_metadata_then_data(context, ch, metadata_out, data_out),
    ]
    run_actor_graph(nodes=nodes, py_executor=py_executor)

    assert metadata_out == [30, 31]
    assert data_out == []


def test_producer_raises_after_metadata(
    context: Context, py_executor: ThreadPoolExecutor
) -> None:
    ch: Channel[ArbitraryChunk[int]] = context.create_channel()
    metadata_out: list[int] = []
    data_out: list[int] = []

    @define_actor()
    async def throwing_producer(
        ctx: Context, ch_out: Channel[ArbitraryChunk[int]]
    ) -> None:
        await ch_out.send_metadata(ctx, Message(0, ArbitraryChunk(99)))
        raise RuntimeError("producer failed")

    nodes: list[PyActor] = [
        throwing_producer(context, ch),
        consume_metadata_then_data(context, ch, metadata_out, data_out),
    ]
    with pytest.raises(RuntimeError, match="producer failed"):
        run_actor_graph(nodes=nodes, py_executor=py_executor)


def test_consumer_raises_with_metadata(
    context: Context, py_executor: ThreadPoolExecutor
) -> None:
    ch: Channel[ArbitraryChunk[int]] = context.create_channel()

    @define_actor()
    async def throwing_consumer(
        ctx: Context, ch_in: Channel[ArbitraryChunk[int]]
    ) -> None:
        raise RuntimeError("consumer failed")

    nodes: list[PyActor] = [
        send_metadata_then_data(context, ch, [1], 0, 1),
        throwing_consumer(context, ch),
    ]
    with pytest.raises(RuntimeError, match="consumer failed"):
        run_actor_graph(nodes=nodes, py_executor=py_executor)
