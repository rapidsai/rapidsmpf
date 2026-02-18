# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Self

from rapidsmpf.streaming.core.actor import CppActor
from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

class BloomFilterChunk:
    # Note: if you go looking for this type in the cython bindings, you
    # won't find it. This is purely to provide for better type-checking of
    # the generic Channel argument to BloomFilter.build/apply below.
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
    ) -> CppActor: ...
    def apply(
        self,
        bloom_filter: Channel[BloomFilterChunk],
        ch_in: Channel[TableChunk],
        ch_out: Channel[TableChunk],
        keys: Iterable[int],
    ) -> CppActor: ...
