# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.node import CppNode
from rapidsmpf.streaming.cudf.partition_chunk import (
    PartitionMapChunk,
    PartitionVectorChunk,
)

def shuffler(
    ctx: Context,
    ch_in: Channel[PartitionMapChunk],
    ch_out: Channel[PartitionVectorChunk],
    op_id: int,
    total_num_partitions: int,
) -> CppNode: ...
