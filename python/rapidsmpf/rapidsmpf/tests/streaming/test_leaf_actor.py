# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from rapidsmpf.streaming.chunks.arbitrary import ArbitraryChunk
from rapidsmpf.streaming.core.actor import run_actor_network
from rapidsmpf.streaming.core.leaf_actor import pull_from_channel, push_to_channel
from rapidsmpf.streaming.core.message import Message

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context


def test_roundtrip(context: Context) -> None:
    expects = [(seq, seq * 2, seq * 3) for seq in range(10)]
    chunks = [
        Message(seq, ArbitraryChunk(expect)) for seq, expect in enumerate(expects)
    ]
    ch1: Channel[ArbitraryChunk[tuple[int, int, int]]] = context.create_channel()
    actor1 = push_to_channel(context, ch_out=ch1, messages=chunks)
    actor2, output = pull_from_channel(context, ch_in=ch1)
    run_actor_network(context, actors=(actor1, actor2))

    results = output.release()
    for seq, (result, expect) in enumerate(zip(results, expects, strict=True)):
        assert result.sequence_number == seq
        assert ArbitraryChunk.from_message(result).release() == expect
