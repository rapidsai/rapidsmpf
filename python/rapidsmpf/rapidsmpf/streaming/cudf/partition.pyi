# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.node import CppNode
from rapidsmpf.streaming.cudf.partition_chunk import (
    PartitionMapChunk,
    PartitionVectorChunk,
)
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

def partition_and_pack(
    ctx: Context,
    ch_in: Channel[TableChunk],
    ch_out: Channel[PartitionMapChunk],
    columns_to_hash: Iterable[int],
    num_partitions: int,
) -> CppNode: ...
def unpack_and_concat(
    ctx: Context,
    ch_in: Channel[PartitionMapChunk]
    | Channel[PartitionVectorChunk]
    | Channel[PartitionMapChunk | PartitionVectorChunk],
    ch_out: Channel[TableChunk],
) -> CppNode: ...
