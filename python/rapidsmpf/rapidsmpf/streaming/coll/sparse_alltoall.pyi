# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.core.context import Context

class SparseAlltoall:
    def __init__(
        self,
        ctx: Context,
        comm: Communicator,
        op_id: int,
        srcs: Iterable[int],
        dsts: Iterable[int],
    ) -> None: ...
    @property
    def comm(self) -> Communicator: ...
    def insert(self, dst: int, packed_data: PackedData) -> None: ...
    async def insert_finished(self, ctx: Context) -> None: ...
    def extract(self, src: int) -> list[PackedData]: ...
