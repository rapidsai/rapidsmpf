# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.expressions import Expression
from pylibcudf.io.parquet import ParquetReaderOptions
from rmm.pylibrmm.stream import Stream

from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.node import CppNode
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

class Filter:
    def __init__(self, stream: Stream, expression: Expression) -> None: ...

def read_parquet(
    ctx: Context,
    ch_out: Channel[TableChunk],
    num_producers: int,
    options: ParquetReaderOptions,
    num_rows_per_chunk: int,
    filter: Filter | None = None,
) -> CppNode: ...
