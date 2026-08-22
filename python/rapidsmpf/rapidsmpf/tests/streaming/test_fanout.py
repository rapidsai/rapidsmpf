# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for streaming fanout actor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rapidsmpf.streaming.chunks.packed_data import PackedDataChunk
from rapidsmpf.streaming.core.actor import run_actor_network
from rapidsmpf.streaming.core.fanout import FanoutPolicy, fanout
from rapidsmpf.streaming.core.leaf_actor import pull_from_channel, push_to_channel
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.testing import generate_packed_data, validate_packed_data

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context


def _message(
    context: Context, sequence_number: int, n_elements: int = 3
) -> Message[PackedDataChunk]:
    stream = context.br().stream_pool.get_stream()
    chunk = PackedDataChunk.from_packed_data(
        generate_packed_data(
            n_elements,
            sequence_number * 10,
            stream,
            context.br(),
        ),
        br=context.br(),
    )
    return Message(sequence_number, chunk)


def _validate(
    context: Context,
    msg: Message[PackedDataChunk],
    sequence_number: int,
    n_elements: int = 3,
) -> None:
    assert msg.sequence_number == sequence_number
    packed = PackedDataChunk.from_message(msg, br=context.br()).to_packed_data()
    validate_packed_data(packed, n_elements, sequence_number * 10)


@pytest.mark.parametrize("policy", [FanoutPolicy.BOUNDED, FanoutPolicy.UNBOUNDED])
def test_fanout_basic(context: Context, policy: FanoutPolicy) -> None:
    """Test basic fanout functionality with multiple output channels."""
    ch_in: Channel[PackedDataChunk] = context.create_channel()
    ch_out1: Channel[PackedDataChunk] = context.create_channel()
    ch_out2: Channel[PackedDataChunk] = context.create_channel()

    messages = [_message(context, i) for i in range(5)]

    push_actor = push_to_channel(context, ch_in, messages)
    fanout_actor = fanout(context, ch_in, [ch_out1, ch_out2], policy)
    pull_actor1, output1 = pull_from_channel(context, ch_out1)
    pull_actor2, output2 = pull_from_channel(context, ch_out2)

    run_actor_network(
        context,
        actors=[push_actor, fanout_actor, pull_actor1, pull_actor2],
    )

    results1 = output1.release()
    results2 = output2.release()

    assert len(results1) == 5, f"Expected 5 messages in output1, got {len(results1)}"
    assert len(results2) == 5, f"Expected 5 messages in output2, got {len(results2)}"

    for i in range(5):
        _validate(context, results1[i], i)
        _validate(context, results2[i], i)


@pytest.mark.parametrize("num_outputs", [1, 3, 5])
@pytest.mark.parametrize("policy", [FanoutPolicy.BOUNDED, FanoutPolicy.UNBOUNDED])
def test_fanout_multiple_outputs(
    context: Context, num_outputs: int, policy: FanoutPolicy
) -> None:
    """Test fanout with varying numbers of output channels."""
    ch_in: Channel[PackedDataChunk] = context.create_channel()
    chs_out: list[Channel[PackedDataChunk]] = [
        context.create_channel() for _ in range(num_outputs)
    ]

    if num_outputs == 1:
        with pytest.raises(ValueError):
            fanout(context, ch_in, chs_out, policy)
        return

    messages = [_message(context, i, n_elements=2) for i in range(3)]

    push_actor = push_to_channel(context, ch_in, messages)
    fanout_actor = fanout(context, ch_in, chs_out, policy)
    pull_actors = []
    outputs = []
    for ch_out in chs_out:
        pull_actor, output = pull_from_channel(context, ch_out)
        pull_actors.append(pull_actor)
        outputs.append(output)

    run_actor_network(
        context,
        actors=[push_actor, fanout_actor, *pull_actors],
    )

    for output_idx, output in enumerate(outputs):
        results = output.release()
        assert len(results) == 3, (
            f"Output {output_idx}: Expected 3 messages, got {len(results)}"
        )
        for i in range(3):
            _validate(context, results[i], i, n_elements=2)


def test_fanout_empty_outputs(context: Context) -> None:
    """Test fanout with empty output list raises value error."""
    ch_in: Channel[PackedDataChunk] = context.create_channel()
    with pytest.raises(ValueError):
        fanout(context, ch_in, [], FanoutPolicy.BOUNDED)


def test_fanout_policy_enum() -> None:
    """Test that FanoutPolicy enum has correct values."""
    assert FanoutPolicy.BOUNDED == 0
    assert FanoutPolicy.UNBOUNDED == 1
    assert len(FanoutPolicy) == 2
