# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import pylibcudf as plc

from rapidsmpf.integrations.cudf.partition import unpack_and_concat
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.chunks.packed_data import PackedDataChunk
from rapidsmpf.streaming.coll.allgather import AllGather, allgather
from rapidsmpf.streaming.core.actor import define_actor, run_actor_network
from rapidsmpf.streaming.core.leaf_actor import pull_from_channel, push_to_channel
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.table_chunk import TableChunk
from rapidsmpf.testing import assert_eq

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    from rapidsmpf.streaming.core.actor import CppActor, PyActor
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context


def test_allgather_actor(context: Context) -> None:
    if context.comm().nranks != 1:
        pytest.skip("Only support single-rank runs")

    num_rows = 1000
    op_id = 0
    stream = context.get_stream_from_pool()
    input_tables = [
        plc.Table(
            [
                plc.Column.from_array(
                    np.arange(num_rows, dtype=np.int32) + i * num_rows, stream=stream
                )
            ]
        )
        for i in range(3)
    ]
    inputs = [
        PackedDataChunk.from_packed_data(
            PackedData.from_cudf_packed_columns(
                plc.contiguous_split.pack(table, stream=stream),
                stream,
                context.br(),
            )
        )
        for table in input_tables
    ]
    actors = []

    ch1: Channel[PackedDataChunk] = context.create_channel()
    actors.append(
        push_to_channel(
            context, ch1, [Message(i, chunk) for i, chunk in enumerate(inputs)]
        )
    )

    ch2: Channel[PackedDataChunk] = context.create_channel()
    actors.append(allgather(context, ch1, ch2, op_id, ordered=True))

    actor, deferred = pull_from_channel(context, ch2)
    actors.append(actor)
    run_actor_network(actors=actors)

    result = unpack_and_concat(
        (
            PackedDataChunk.from_message(msg).to_packed_data()
            for msg in deferred.release()
        ),
        stream,
        context.br(),
    )

    expect = plc.concatenate.concatenate(input_tables, stream=stream)
    stream.synchronize()
    assert_eq(result, expect)


@define_actor()
async def generate_inputs(
    context: Context, ch: Channel[PackedDataChunk], num_rows: int, num_chunks: int
) -> None:
    for i in range(num_chunks):
        stream = context.get_stream_from_pool()
        table = plc.Table(
            [
                plc.Column.from_array(
                    np.arange(num_rows, dtype=np.int32) + i * num_rows, stream=stream
                )
            ]
        )
        msg = Message(
            i,
            PackedDataChunk.from_packed_data(
                PackedData.from_cudf_packed_columns(
                    plc.contiguous_split.pack(table, stream=stream),
                    stream,
                    context.br(),
                )
            ),
        )
        await ch.send(context, msg)
    await ch.drain(context)


@define_actor()
async def allgather_and_concat(
    context: Context,
    ch_in: Channel[PackedDataChunk],
    ch_out: Channel[TableChunk],
    op_id: int,
) -> None:
    gather = AllGather(context, op_id)
    while (msg := await ch_in.recv(context)) is not None:
        chunk = PackedDataChunk.from_message(msg).to_packed_data()
        gather.insert(msg.sequence_number, chunk)
    gather.insert_finished()
    gathered = await gather.extract_all(context, ordered=True)
    stream = context.get_stream_from_pool()
    table = unpack_and_concat(gathered, stream, context.br())
    to_send = TableChunk.from_pylibcudf_table(table, stream, exclusive_view=True)
    await ch_out.send(context, Message(0, to_send))
    await ch_out.drain(context)


def test_allgather_object_interface(
    context: Context, py_executor: ThreadPoolExecutor
) -> None:
    ch_in: Channel[PackedDataChunk] = context.create_channel()
    ch_out: Channel[TableChunk] = context.create_channel()
    actors: list[CppActor | PyActor] = []
    num_rows = 100
    num_chunks = 10
    op_id = 0
    actors.append(generate_inputs(context, ch_in, num_rows, num_chunks))
    actors.append(allgather_and_concat(context, ch_in, ch_out, op_id))

    actor, deferred = pull_from_channel(context, ch_out)
    actors.append(actor)

    run_actor_network(actors=actors, py_executor=py_executor)
    (result_msg,) = deferred.release()
    result = TableChunk.from_message(result_msg)
    expect = plc.Table(
        [
            plc.Column.from_array(
                np.arange(num_rows * num_chunks, dtype=np.int32), stream=result.stream
            )
        ]
    )
    got = result.table_view()
    result.stream.synchronize()
    assert_eq(expect, got)
