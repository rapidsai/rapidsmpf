# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping

from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.shuffler import PartitionAssignment
from rapidsmpf.streaming.chunks.partition import PartitionMapChunk, PartitionVectorChunk
from rapidsmpf.streaming.core.actor import CppActor
from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context

def shuffler(
    ctx: Context,
    comm: Communicator,
    ch_in: Channel[PartitionMapChunk],
    ch_out: Channel[PartitionVectorChunk],
    op_id: int,
    total_num_partitions: int,
    partition_assignment: PartitionAssignment = ...,
) -> CppActor: ...

class ShufflerAsync:
    def __init__(
        self,
        ctx: Context,
        comm: Communicator,
        op_id: int,
        total_num_partitions: int,
        partition_assignment: PartitionAssignment = ...,
    ) -> None: ...
    @property
    def comm(self) -> Communicator: ...
    def insert(self, chunks: Mapping[int, PackedData]) -> None: ...
    async def insert_finished(self, ctx: Context) -> None: ...
    def extract(self, pid: int) -> list[PackedData]: ...
    def local_partitions(self) -> list[int]: ...
