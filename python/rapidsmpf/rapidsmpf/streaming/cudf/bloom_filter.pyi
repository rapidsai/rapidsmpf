# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Self

from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import CppNode
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

class BloomFilterChunk:
    @classmethod
    def from_message(cls: type[Self], message: Message[Self]) -> Self: ...
    def into_message(self, sequence_number: int, message: Message[Self]) -> None: ...

class BloomFilter:
    def __init__(self, ctx: Context, seed: int, num_filter_blocks: int) -> None: ...
    @staticmethod
    def fitting_num_blocks(l2size: int) -> int: ...
    def build(
        self,
        ch_in: Channel[TableChunk],
        ch_out: Channel[BloomFilterChunk],
        tag: int,
    ) -> CppNode: ...
    def apply(
        self,
        bloom_filter: Channel[BloomFilterChunk],
        ch_in: Channel[TableChunk],
        ch_out: Channel[TableChunk],
        keys: Iterable[int],
    ) -> CppNode: ...
