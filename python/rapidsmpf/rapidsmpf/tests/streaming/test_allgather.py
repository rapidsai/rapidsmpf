# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import pytest

from rapidsmpf.streaming.chunks.packed_data import PackedDataChunk
from rapidsmpf.streaming.coll.allgather import AllGather, allgather
from rapidsmpf.streaming.core.actor import define_actor, run_actor_network
from rapidsmpf.streaming.core.leaf_actor import pull_from_channel, push_to_channel
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.testing import generate_packed_data, validate_packed_data

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.actor import CppActor
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context


def _make_chunk(context: Context, num_rows: int, offset: int) -> PackedDataChunk:
    stream = context.br().stream_pool.get_stream()
    return PackedDataChunk.from_packed_data(
        generate_packed_data(num_rows, offset, stream, context.br()),
        br=context.br(),
    )


def _validate_message(
    context: Context, msg: Message[PackedDataChunk], num_rows: int, offset: int
) -> None:
    packed = PackedDataChunk.from_message(msg, br=context.br()).to_packed_data()
    validate_packed_data(packed, num_rows, offset)


def test_allgather_actor(context: Context, comm: Communicator) -> None:
    if comm.nranks != 1:
        pytest.skip("Only support single-rank runs")

    num_rows = 1000
    op_id = 0
    inputs = [_make_chunk(context, num_rows, i * num_rows) for i in range(3)]
    actors = []

    ch1: Channel[PackedDataChunk] = context.create_channel()
    actors.append(
        push_to_channel(
            context, ch1, [Message(i, chunk) for i, chunk in enumerate(inputs)]
        )
    )

    ch2: Channel[PackedDataChunk] = context.create_channel()
    actors.append(allgather(context, comm, ch1, ch2, op_id, ordered=True))

    actor, deferred = pull_from_channel(context, ch2)
    actors.append(actor)
    run_actor_network(context, actors=actors)

    results = deferred.release()
    assert len(results) == len(inputs)
    for i, msg in enumerate(results):
        assert msg.sequence_number == i
        _validate_message(context, msg, num_rows, i * num_rows)


@define_actor()
async def generate_inputs(
    context: Context, ch: Channel[PackedDataChunk], num_rows: int, num_chunks: int
) -> None:
    for i in range(num_chunks):
        await ch.send(context, Message(i, _make_chunk(context, num_rows, i * num_rows)))
    await ch.drain(context)


@define_actor()
async def allgather_and_forward(
    context: Context,
    comm: Communicator,
    ch_in: Channel[PackedDataChunk],
    ch_out: Channel[PackedDataChunk],
    op_id: int,
    use_context_manager: bool,  # noqa: FBT001
) -> None:
    gather = AllGather(context, comm, op_id)
    cm = gather if use_context_manager else nullcontext(gather)
    with cm as ag:
        while (msg := await ch_in.recv(context)) is not None:
            chunk = PackedDataChunk.from_message(msg, br=context.br()).to_packed_data()
            ag.insert(msg.sequence_number, chunk)
    if not use_context_manager:
        gather.insert_finished()
    gathered = await gather.extract_all(context, ordered=True)
    for sequence, packed in enumerate(gathered):
        await ch_out.send(
            context,
            Message(
                sequence,
                PackedDataChunk.from_packed_data(packed, br=context.br()),
            ),
        )
    await ch_out.drain(context)


@pytest.mark.parametrize(
    "use_context_manager", [True, False], ids=["context", "non-context"]
)
def test_allgather_object_interface(
    context: Context,
    comm: Communicator,
    use_context_manager: bool,  # noqa: FBT001
) -> None:
    if comm.nranks != 1:
        pytest.skip("Only support single-rank runs")

    ch_in: Channel[PackedDataChunk] = context.create_channel()
    ch_out: Channel[PackedDataChunk] = context.create_channel()
    actors: list[CppActor | Awaitable[None]] = []
    num_rows = 100
    num_chunks = 10
    op_id = 0
    actors.append(generate_inputs(context, ch_in, num_rows, num_chunks))
    actors.append(
        allgather_and_forward(context, comm, ch_in, ch_out, op_id, use_context_manager)
    )

    actor, deferred = pull_from_channel(context, ch_out)
    actors.append(actor)

    run_actor_network(context, actors=actors)
    results = deferred.release()
    assert len(results) == num_chunks
    for i, msg in enumerate(results):
        assert msg.sequence_number == i
        _validate_message(context, msg, num_rows, i * num_rows)
