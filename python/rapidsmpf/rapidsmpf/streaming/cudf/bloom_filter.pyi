# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.node import CppNode
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

class BloomFilter:
    def __init__(self, ctx: Context, seed: int, num_filter_blocks: int) -> None: ...
    @staticmethod
    def fitting_num_blocks(l2size: int) -> int: ...
    def build(
        self,
        ch_in: Channel[TableChunk],
        ch_out: Channel,
        tag: int,
    ) -> CppNode: ...
    def apply(
        self,
        bloom_filter: Channel,
        ch_in: Channel[TableChunk],
        ch_out: Channel[TableChunk],
        keys: Iterable[int],
    ) -> CppNode: ...
