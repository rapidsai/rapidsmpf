# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rapidsmpf.streaming.chunks.arbitrary import ArbitraryChunk
from rapidsmpf.streaming.core.actor import define_actor, run_actor_network
from rapidsmpf.streaming.core.leaf_actor import pull_from_channel, push_to_channel
from rapidsmpf.streaming.core.message import Message


@pytest.fixture
def expects() -> list[tuple[int, int, int]]:
    return [(seq, seq * 2, seq * 3) for seq in range(10)]


if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context


def test_send_arbitrary_chunks(
    context: Context, expects: list[tuple[int, int, int]]
) -> None:
    ch1: Channel[ArbitraryChunk[tuple[int, int, int]]] = context.create_channel()

    # The actor accesses `ch1` both through the `ch_out` parameter and the closure.
    @define_actor(extra_channels=(ch1,))
    async def actor1(ctx: Context, /, ch_out: Channel) -> None:
        for seq, chunk in enumerate(expects):
            await ch1.send(ctx, Message(seq, ArbitraryChunk(chunk)))
        await ch_out.drain(ctx)

    actor2, output = pull_from_channel(context, ch_in=ch1)

    run_actor_network(
        context,
        actors=[
            actor1(context, ch_out=ch1),
            actor2,
        ],
    )

    results = output.release()
    for seq, (result, expect) in enumerate(zip(results, expects, strict=True)):
        assert result.sequence_number == seq
        assert ArbitraryChunk.from_message(result).release() == expect


def test_shutdown(context: Context) -> None:
    @define_actor()
    async def actor1(ctx: Context, ch_out: Channel[ArbitraryChunk[int]]) -> None:
        await ch_out.shutdown(ctx)
        # Calling shutdown multiple times is allowed.
        await ch_out.shutdown(ctx)

    ch1: Channel[ArbitraryChunk[int]] = context.create_channel()
    actor2, output = pull_from_channel(context, ch_in=ch1)

    run_actor_network(
        context,
        actors=[
            actor1(context, ch_out=ch1),
            actor2,
        ],
    )

    assert output.release() == []


def test_send_error(context: Context) -> None:
    @define_actor()
    async def actor1(ctx: Context, ch_out: Channel[ArbitraryChunk[int]]) -> None:
        raise RuntimeError("MyError")

    ch1: Channel[ArbitraryChunk[int]] = context.create_channel()
    actor2, output = pull_from_channel(context, ch_in=ch1)

    with pytest.RaisesGroup(
        pytest.RaisesExc(
            RuntimeError,
            match="MyError",
        )
    ):
        run_actor_network(
            context,
            actors=[
                actor1(context, ch_out=ch1),
                actor2,
            ],
        )

    assert output.release() == []


def test_recv_arbitrary_chunks(
    context: Context, expects: list[tuple[int, int, int]]
) -> None:
    chunks = [
        Message(seq, ArbitraryChunk(expect)) for seq, expect in enumerate(expects)
    ]

    results: list[Message[ArbitraryChunk[tuple[int, int, int]]]] = []

    @define_actor()
    async def actor1(
        ctx: Context, ch_in: Channel[ArbitraryChunk[tuple[int, int, int]]]
    ) -> None:
        while True:
            chunk = await ch_in.recv(ctx)
            if chunk is None:
                break
            results.append(chunk)

    ch1: Channel[ArbitraryChunk[tuple[int, int, int]]] = context.create_channel()

    run_actor_network(
        context,
        actors=[
            push_to_channel(context, ch_out=ch1, messages=chunks),
            actor1(context, ch_in=ch1),
        ],
    )

    for seq, (result, expect) in enumerate(zip(results, expects, strict=True)):
        assert result.sequence_number == seq
        assert ArbitraryChunk.from_message(result).release() == expect


@pytest.mark.filterwarnings("error")
def test_unawaited_actor_closed_coroutines_no_warning(context: Context) -> None:
    ch: Channel[ArbitraryChunk[int]] = context.create_channel()

    @define_actor()
    async def my_actor(ctx: Context, ch_out: Channel[ArbitraryChunk[int]]) -> None:
        await ch_out.send(ctx, Message(0, ArbitraryChunk(42)))
        await ch_out.drain(ctx)

    # Never awaited, just verifying no RuntimeWarning is emitted
    actor = my_actor(context, ch_out=ch)
    del actor
