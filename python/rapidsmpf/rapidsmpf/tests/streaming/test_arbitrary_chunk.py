# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

from rapidsmpf.streaming.chunks.arbitrary import ArbitraryChunk
from rapidsmpf.streaming.core.actor import define_actor, run_actor_graph
from rapidsmpf.streaming.core.leaf_actor import pull_from_channel, push_to_channel
from rapidsmpf.streaming.core.message import Message

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor
    from typing import Any

    from rapidsmpf.streaming.core.actor import CppActor, PyActor
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context


class Object:
    def __init__(self, value: Any):
        self.value = value


def test_roundtrip_chunk(context: Context) -> None:
    expect = Object(10)
    got = ArbitraryChunk(expect).release()
    assert got is expect


def test_roundtrip_message() -> None:
    expect = Object(10)
    got = ArbitraryChunk.from_message(Message(1, ArbitraryChunk(expect))).release()
    assert got is expect


def test_gc_in_chunk() -> None:
    obj = Object(10)
    finalizer = weakref.finalize(obj, lambda: None)
    assert finalizer.alive

    chunk = ArbitraryChunk(obj)
    del obj
    assert finalizer.alive
    del chunk
    assert not finalizer.alive


def test_gc_in_message() -> None:
    obj = Object(10)
    finalizer = weakref.finalize(obj, lambda: None)
    assert finalizer.alive

    chunk = ArbitraryChunk(obj)
    del obj
    assert finalizer.alive
    message = Message(1, chunk)
    del chunk
    assert finalizer.alive
    del message
    assert not finalizer.alive


def test_gc_after_message_release() -> None:
    obj = Object(10)
    finalizer = weakref.finalize(obj, lambda: None)
    assert finalizer.alive

    chunk = ArbitraryChunk(obj)
    del obj
    assert finalizer.alive
    message = Message(1, chunk)
    del chunk
    assert finalizer.alive
    chunk = ArbitraryChunk.from_message(message)
    del message
    assert finalizer.alive
    del chunk
    assert not finalizer.alive


def test_gc_after_chunk_release() -> None:
    obj = Object(10)
    addr = id(obj)
    finalizer = weakref.finalize(obj, lambda: None)
    assert finalizer.alive

    chunk = ArbitraryChunk(obj)
    del obj
    assert finalizer.alive
    message = Message(1, chunk)
    del chunk
    assert finalizer.alive
    chunk = ArbitraryChunk.from_message(message)
    del message
    assert finalizer.alive
    obj = chunk.release()
    del chunk
    assert finalizer.alive
    assert id(obj) == addr
    assert obj.value == 10
    del obj
    assert not finalizer.alive


def test_with_channel(context: Context, py_executor: ThreadPoolExecutor) -> None:
    ch_in: Channel[ArbitraryChunk[int]] = context.create_channel()
    ch_out: Channel[ArbitraryChunk[int]] = context.create_channel()

    inputs = [Message(seq, ArbitraryChunk(seq)) for seq in range(10)]

    @define_actor()
    async def increment(
        ctx: Context,
        ch_in: Channel[ArbitraryChunk[int]],
        ch_out: Channel[ArbitraryChunk[int]],
    ) -> None:
        while (msg := await ch_in.recv(ctx)) is not None:
            value = ArbitraryChunk.from_message(msg).release()
            assert value == msg.sequence_number

            await ch_out.send(
                ctx, Message(msg.sequence_number, ArbitraryChunk(value + 1))
            )
        await ch_out.drain(ctx)

    nodes: list[CppActor | PyActor] = [
        push_to_channel(context, ch_in, inputs),
        increment(context, ch_in, ch_out),
    ]
    node, deferred_messages = pull_from_channel(context, ch_out)
    nodes.append(node)

    run_actor_graph(nodes=nodes, py_executor=py_executor)

    results = deferred_messages.release()
    for seq, msg in enumerate(results):
        value = ArbitraryChunk.from_message(msg).release()
        assert isinstance(value, int)
        assert value == msg.sequence_number + 1
        assert seq == msg.sequence_number
