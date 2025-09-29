# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import pytest

import cudf

from rapidsmpf.streaming.coll.shuffler import shuffler
from rapidsmpf.streaming.core.channel import Channel, Message
from rapidsmpf.streaming.core.leaf_node import pull_from_channel, push_to_channel
from rapidsmpf.streaming.core.node import run_streaming_pipeline
from rapidsmpf.streaming.cudf.partition import partition_and_pack, unpack_and_concat
from rapidsmpf.streaming.cudf.table_chunk import TableChunk
from rapidsmpf.testing import assert_eq
from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table, pylibcudf_to_cudf_dataframe

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.cudf.partition_chunk import (
        PartitionMapChunk,
        PartitionVectorChunk,
    )


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
            sequence_number=i,
            table=cudf_to_pylibcudf_table(df_chunk),
            stream=stream,
            exclusive_view=False,
        )
        input_chunks.append(Message(chunk))

    # Build the streaming pipeline:
    #   push -> partition/pack -> shuffle -> unpack/concat -> pull.
    nodes = []

    ch1: Channel[TableChunk] = Channel()
    nodes.append(push_to_channel(context, ch1, input_chunks))

    ch2: Channel[PartitionMapChunk] = Channel()
    nodes.append(
        partition_and_pack(
            context,
            ch_in=ch1,
            ch_out=ch2,
            columns_to_hash=(df.columns.get_loc("val"),),
            num_partitions=num_partitions,
        )
    )

    ch3: Channel[PartitionVectorChunk] = Channel()
    nodes.append(
        shuffler(
            context,
            ch_in=ch2,
            ch_out=ch3,
            op_id=op_id,
            total_num_partitions=num_partitions,
        )
    )

    ch4: Channel[TableChunk] = Channel()
    nodes.append(unpack_and_concat(context, ch_in=ch3, ch_out=ch4))

    pull_node, out_messages = pull_from_channel(context, ch_in=ch4)
    nodes.append(pull_node)

    # Run all nodes. This blocks until every node has completed.
    run_streaming_pipeline(nodes=nodes)

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
