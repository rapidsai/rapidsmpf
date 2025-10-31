# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

from rapidsmpf.streaming.chunks.arbitrary import ArbitraryChunk
from rapidsmpf.streaming.core.message import Message

if TYPE_CHECKING:
    from typing import Any

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
