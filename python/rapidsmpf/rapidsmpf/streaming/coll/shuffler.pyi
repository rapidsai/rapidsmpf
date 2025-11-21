# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping

from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.chunks.partition import PartitionMapChunk, PartitionVectorChunk
from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.node import CppNode

def shuffler(
    ctx: Context,
    ch_in: Channel[PartitionMapChunk],
    ch_out: Channel[PartitionVectorChunk],
    op_id: int,
    total_num_partitions: int,
) -> CppNode: ...

class ShufflerAsync:
    def __init__(self, ctx: Context, op_id: int, total_num_partitions: int) -> None: ...
    def insert(self, chunks: Mapping[int, PackedData]) -> None: ...
    async def insert_finished(self, ctx: Context) -> None: ...
    async def extract_async(
        self, ctx: Context, pid: int
    ) -> list[PackedData] | None: ...
    async def extract_any_async(
        self, ctx: Context
    ) -> tuple[int, list[PackedData]] | None: ...
