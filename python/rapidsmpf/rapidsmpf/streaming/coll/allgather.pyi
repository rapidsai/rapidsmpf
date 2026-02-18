# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.chunks.packed_data import PackedDataChunk
from rapidsmpf.streaming.core.actor import CppActor
from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context

class AllGather:
    def __init__(self, ctx: Context, op_id: int) -> None: ...
    def insert(self, sequence_number: int, packed_data: PackedData) -> None: ...
    def insert_finished(self) -> None: ...
    async def extract_all(self, ctx: Context, *, ordered: bool) -> list[PackedData]: ...

def allgather(
    ctx: Context,
    ch_in: Channel[PackedDataChunk],
    ch_out: Channel[PackedDataChunk],
    op_id: int,
    *,
    ordered: bool,
) -> CppActor: ...
