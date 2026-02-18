# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import cudf

from rapidsmpf.streaming.core.actor import run_actor_graph
from rapidsmpf.streaming.core.leaf_actor import pull_from_channel, push_to_channel
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.partition import partition_and_pack, unpack_and_concat
from rapidsmpf.streaming.cudf.table_chunk import TableChunk
from rapidsmpf.testing import assert_eq
from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.cudf.partition_chunk import PartitionMapChunk


@pytest.mark.parametrize("num_partitions", [1, 2, 3, 10])
def test_partition_and_pack_unpack(
    context: Context, stream: Stream, num_partitions: int
) -> None:
    expects = [
        cudf_to_pylibcudf_table(cudf.DataFrame({"0": [1, 2, 3], "1": [2, 2, 1]})),
        cudf_to_pylibcudf_table(cudf.DataFrame({"0": [], "1": []})),
    ]
    table_chunks = [
        Message(
            seq, TableChunk.from_pylibcudf_table(expect, stream, exclusive_view=False)
        )
        for seq, expect in enumerate(expects)
    ]
    ch1: Channel[TableChunk] = context.create_channel()
    actor1 = push_to_channel(context, ch_out=ch1, messages=table_chunks)

    ch2: Channel[PartitionMapChunk] = context.create_channel()
    actor2 = partition_and_pack(
        context,
        ch_in=ch1,
        ch_out=ch2,
        columns_to_hash=(1,),
        num_partitions=num_partitions,
    )

    ch3: Channel[TableChunk] = context.create_channel()
    actor3 = unpack_and_concat(
        context,
        ch_in=ch2,
        ch_out=ch3,
    )

    actor4, output = pull_from_channel(context, ch_in=ch3)
    run_actor_graph(actors=(actor1, actor2, actor3, actor4))

    results = output.release()
    for seq, (result, expect) in enumerate(zip(results, expects, strict=True)):
        assert result.sequence_number == seq
        tbl = TableChunk.from_message(result)
        assert_eq(tbl.table_view(), expect, sort_rows="0")
