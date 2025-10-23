# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

import pytest

from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.leaf_node import pull_from_channel, push_to_channel
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import define_py_node, run_streaming_pipeline
from rapidsmpf.streaming.core.pyobject import PyObjectPayload

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    from rapidsmpf.streaming.core.context import Context


@pytest.mark.parametrize(
    "data",
    [
        42,
        4.2,
        "hello",
        False,
        None,
        {"key": "value", "number": 42},
        [1, 2, 3, 4, 5],
        ("a", "b", "c"),
    ],
)
def test_from_object_basic(data: Any) -> None:
    """Test creating PyObjectPayload from basic Python objects."""
    payload = PyObjectPayload.from_object(sequence_number=0, obj=data)
    assert payload.sequence_number == 0
    assert payload.extract_object() == data

    payload = PyObjectPayload.from_object(sequence_number=18, obj=data)
    assert payload.sequence_number == 18
    assert payload.extract_object() == data


def test_message_roundtrip() -> None:
    """Test that PyObjectPayload can be wrapped in a message and extracted."""
    original_data = {"user": "alice", "score": 100, "active": True}
    payload1 = PyObjectPayload.from_object(sequence_number=42, obj=original_data)

    # Wrap in message
    msg = Message(payload1)
    assert msg.empty() is False

    # Extract from message
    payload2 = PyObjectPayload.from_message(msg)
    assert msg.empty() is True  # Message should be empty after extraction

    # Verify data
    assert payload2.sequence_number == 42
    assert payload2.extract_object() == original_data


def test_streaming_pipeline(context: Context, py_executor: ThreadPoolExecutor) -> None:
    """Test PyObjectPayload in a basic streaming pipeline."""
    # Create test data
    test_objects = [
        {"name": "alice", "value": 10},
        {"name": "bob", "value": 20},
        {"name": "charlie", "value": 30},
    ]

    # Wrap in messages
    messages = [
        Message(PyObjectPayload.from_object(seq, obj))
        for seq, obj in enumerate(test_objects)
    ]

    # Create channels
    ch_in: Channel[PyObjectPayload] = Channel()
    ch_out: Channel[PyObjectPayload] = Channel()

    # Producer node
    producer = push_to_channel(context, ch_out=ch_in, messages=messages)

    # Transform node - multiply values by 10
    @define_py_node()
    async def multiply_values(ctx: Context, ch_in: Channel, ch_out: Channel) -> None:
        while (msg := await ch_in.recv(ctx)) is not None:
            payload = PyObjectPayload.from_message(msg)
            data = payload.extract_object()
            # Transform data
            data["value"] *= 10
            # Forward transformed data
            new_msg = Message(
                PyObjectPayload.from_object(payload.sequence_number, data)
            )
            await ch_out.send(ctx, new_msg)
        await ch_out.drain(ctx)

    transformer = multiply_values(context, ch_in=ch_in, ch_out=ch_out)

    # Consumer node
    consumer, output = pull_from_channel(context, ch_in=ch_out)

    # Run pipeline
    run_streaming_pipeline(
        nodes=(producer, transformer, consumer),
        py_executor=py_executor,
    )

    # Verify results
    results = []
    for msg in output.release():
        payload = PyObjectPayload.from_message(msg)
        results.append(payload.extract_object())
    assert len(results) == len(test_objects)

    # Values should be multiplied by 10
    assert results[0]["value"] == 100
    assert results[1]["value"] == 200
    assert results[2]["value"] == 300


def test_pyobject_garbage_collection() -> None:
    """Test that wrapped objects are properly garbage collected."""

    class TrackedObject:
        def __init__(self, value: int) -> None:
            self.value = value

    # Create object and attach finalizer
    finalized = []
    obj = TrackedObject(42)
    weakref.finalize(obj, lambda: finalized.append(True))

    # Wrap in payload
    payload1 = PyObjectPayload.from_object(sequence_number=0, obj=obj)

    # Delete original reference - object should still be alive in payload
    del obj
    assert len(finalized) == 0, "Object should not be collected yet"

    # Wrap in message
    msg = Message(payload1)
    del payload1
    assert len(finalized) == 0, "Object should still be alive in message"

    # Extract from message
    payload2 = PyObjectPayload.from_message(msg)
    assert len(finalized) == 0, "Object should still be alive in extracted payload"

    # Extract object (this consumes the payload)
    retrieved_obj = payload2.extract_object()
    assert retrieved_obj.value == 42
    assert len(finalized) == 0, "Object should still be alive while we hold reference"

    # Delete the extracted object - now it should be collected
    del retrieved_obj
    assert len(finalized) == 1, (
        "Object should be collected after all references are gone"
    )
