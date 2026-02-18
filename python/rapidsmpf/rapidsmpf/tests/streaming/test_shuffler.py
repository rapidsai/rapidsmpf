# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import pytest

import cudf
import pylibcudf as plc

from rapidsmpf.integrations.cudf.partition import split_and_pack, unpack_and_concat
from rapidsmpf.streaming.coll.shuffler import ShufflerAsync, shuffler
from rapidsmpf.streaming.core.actor import define_actor, run_actor_graph
from rapidsmpf.streaming.core.leaf_actor import pull_from_channel, push_to_channel
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.partition import (
    partition_and_pack,
    unpack_and_concat as streaming_unpack_and_concat,
)
from rapidsmpf.streaming.cudf.table_chunk import TableChunk
from rapidsmpf.testing import assert_eq
from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table, pylibcudf_to_cudf_dataframe

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.streaming.chunks.partition import (
        PartitionMapChunk,
        PartitionVectorChunk,
    )
    from rapidsmpf.streaming.core.actor import CppActor, PyActor
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context


@pytest.mark.parametrize("num_partitions", [1, 2, 3, 10])
def test_single_rank_shuffler(
    context: Context, stream: Stream, num_partitions: int
) -> None:
    if context.comm().nranks != 1:
        pytest.skip("Only support single-rank runs")

    num_rows = 1000
    num_chunks = 5
    chunk_size = num_rows // num_chunks
    op_id = 0
    # We start a full dataframe.
    df = cudf.DataFrame(
        {
            "idx": cp.arange(num_rows, dtype=cp.int32),
            "val": cp.random.randint(0, 10, size=num_rows, dtype=cp.int32),
        }
    )

    # That we slice into chunks and wrap as TableChunk (sequence_number=i).
    input_chunks: list[Message[TableChunk]] = []
    for i in range(num_chunks):
        lo = i * chunk_size
        hi = (i + 1) * chunk_size
        df_chunk = df.iloc[lo:hi]
        chunk = TableChunk.from_pylibcudf_table(
            table=cudf_to_pylibcudf_table(df_chunk),
            stream=stream,
            exclusive_view=False,
        )
        input_chunks.append(Message(i, chunk))

    # Build the streaming pipeline:
    #   push -> partition/pack -> shuffle -> unpack/concat -> pull.
    nodes = []

    ch1: Channel[TableChunk] = context.create_channel()
    nodes.append(push_to_channel(context, ch1, input_chunks))

    ch2: Channel[PartitionMapChunk] = context.create_channel()
    nodes.append(
        partition_and_pack(
            context,
            ch_in=ch1,
            ch_out=ch2,
            columns_to_hash=(df.columns.get_loc("val"),),
            num_partitions=num_partitions,
        )
    )

    ch3: Channel[PartitionVectorChunk] = context.create_channel()
    nodes.append(
        shuffler(
            context,
            ch_in=ch2,
            ch_out=ch3,
            op_id=op_id,
            total_num_partitions=num_partitions,
        )
    )

    ch4: Channel[TableChunk] = context.create_channel()
    nodes.append(streaming_unpack_and_concat(context, ch_in=ch3, ch_out=ch4))

    pull_node, out_messages = pull_from_channel(context, ch_in=ch4)
    nodes.append(pull_node)

    # Run all nodes. This blocks until every node has completed.
    run_actor_graph(nodes=nodes)

    # Unwrap the messages into table chunks.
    output_chunks = [TableChunk.from_message(msg) for msg in out_messages.release()]

    # Concatenate all output chunks into a single cuDF DataFrame
    result = cudf.concat(
        [
            pylibcudf_to_cudf_dataframe(chunk.table_view(), column_names=df.columns)
            for chunk in output_chunks
        ],
        ignore_index=True,
    )
    assert_eq(result, df, sort_rows="idx")


@define_actor()
async def generate_inputs(
    context: Context, ch: Channel[TableChunk], num_rows: int, num_chunks: int
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
            i, TableChunk.from_pylibcudf_table(table, stream, exclusive_view=True)
        )
        await ch.send(context, msg)
    await ch.drain(context)


@define_actor()
async def do_shuffle(
    context: Context,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    op_id: int,
    num_partitions: int,
    *,
    use_extract_any: bool,
) -> None:
    shuffle = ShufflerAsync(context, op_id, num_partitions)
    while (msg := await ch_in.recv(context)) is not None:
        chunk = TableChunk.from_message(msg)
        num_rows = chunk.table_view().num_rows()
        part_size = num_rows // num_partitions + (num_rows % num_partitions)
        splits = range(part_size, num_rows, part_size)
        shuffle.insert(
            split_and_pack(chunk.table_view(), splits, chunk.stream, context.br())
        )
    await shuffle.insert_finished(context)
    if use_extract_any:
        while (out := await shuffle.extract_any_async(context)) is not None:
            pid, data = out
            stream = context.get_stream_from_pool()
            unpacked = TableChunk.from_pylibcudf_table(
                unpack_and_concat(data, stream, context.br()),
                stream,
                exclusive_view=True,
            )
            await ch_out.send(context, Message(pid, unpacked))
    else:
        # TODO: this is only for a single rank
        for pid in range(num_partitions):
            pd = await shuffle.extract_async(context, pid)
            assert pd is not None
            stream = context.get_stream_from_pool()
            unpacked = TableChunk.from_pylibcudf_table(
                unpack_and_concat(pd, stream, context.br()),
                stream,
                exclusive_view=True,
            )
            await ch_out.send(context, Message(pid, unpacked))
    await ch_out.drain(context)


@pytest.mark.parametrize("use_extract_any", [False, True])
def test_shuffler_object_interface(
    context: Context,
    py_executor: ThreadPoolExecutor,
    use_extract_any: bool,  # noqa: FBT001
) -> None:
    nodes: list[CppActor | PyActor] = []

    num_partitions = 5
    num_rows = 100
    num_chunks = 4
    op_id = 0
    ch_in: Channel[TableChunk] = context.create_channel()
    nodes.append(generate_inputs(context, ch_in, num_rows, num_chunks))
    ch_shuffled: Channel[TableChunk] = context.create_channel()
    nodes.append(
        do_shuffle(
            context,
            ch_in,
            ch_shuffled,
            op_id,
            num_partitions,
            use_extract_any=use_extract_any,
        )
    )
    node, deferred = pull_from_channel(context, ch_shuffled)
    nodes.append(node)

    run_actor_graph(nodes=nodes, py_executor=py_executor)
    messages = deferred.release()
    # TODO: single rank only assertions
    assert len(messages) == 5
    if use_extract_any:
        assert {msg.sequence_number for msg in messages} == set(range(num_partitions))
    else:
        assert [msg.sequence_number for msg in messages] == list(range(num_partitions))
    chunks = [(msg.sequence_number, TableChunk.from_message(msg)) for msg in messages]

    full_column = np.arange(num_rows * num_chunks, dtype=np.int32)
    part_size = num_rows // num_partitions + (num_rows % num_partitions)
    splits = [*range(0, num_rows, part_size), num_rows]
    for pid, table in chunks:
        expect = plc.Column.from_array(
            np.concat(
                [
                    full_column[i * num_rows : (i + 1) * num_rows][
                        splits[pid] : splits[pid + 1]
                    ]
                    for i in range(num_chunks)
                ]
            ),
            stream=table.stream,
        )
        got = table.table_view()
        table.stream.synchronize()
        assert_eq(plc.Table([expect]), got, sort_rows="0")
