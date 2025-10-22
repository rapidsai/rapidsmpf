# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.io.parquet import ParquetReaderOptions

from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.node import CppNode
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

def read_parquet(
    ctx: Context,
    ch_out: Channel[TableChunk],
    max_tickets: int,
    options: ParquetReaderOptions,
    num_rows_per_chunk: int,
) -> CppNode: ...
