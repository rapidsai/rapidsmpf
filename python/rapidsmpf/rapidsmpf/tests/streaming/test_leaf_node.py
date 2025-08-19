# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import cudf

from rapidsmpf.streaming.core.node import run_streaming_pipeline
from rapidsmpf.streaming.cudf.table_chunk import (
    DeferredOutputChunks,
    TableChunk,
    TableChunkChannel,
    pull_chunks_from_channel,
    push_table_chunks_to_channel,
)

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.core.node import Node


def test_roundtrip(context: Context, stream: Stream) -> None:
    expects = [
        cudf.DataFrame({"a": [1 * seq, 2 * seq, 3 * seq]}).to_pylibcudf()[0]
        for seq in range(10)
    ]
    table_chunks = [
        TableChunk.from_pylibcudf_table(seq, expects[seq], stream) for seq in range(10)
    ]
    ch1 = TableChunkChannel()
    nodes: list[Node] = []
    nodes.append(
        push_table_chunks_to_channel(ctx=context, ch_out=ch1, chunks=table_chunks)
    )
    output = DeferredOutputChunks()
    nodes.append(pull_chunks_from_channel(ctx=context, ch_in=ch1, chunks=output))

    run_streaming_pipeline(nodes)
