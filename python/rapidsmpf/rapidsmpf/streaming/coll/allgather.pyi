# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rapidsmpf.streaming.chunks.packed_data import PackedDataChunk
from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.node import CppNode

def allgather(
    ctx: Context,
    ch_in: Channel[PackedDataChunk],
    ch_out: Channel[PackedDataChunk],
    op_id: int,
    *ordered: bool,
) -> CppNode: ...
