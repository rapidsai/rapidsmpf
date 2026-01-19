# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rapidsmpf.config import Options
from rapidsmpf.memory.memory_reservation import MemoryReservation
from rapidsmpf.rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.core.context import Context

class MemoryReserveOrWait:
    def __init__(self, options: Options, mem_type: MemoryType, ctx: Context): ...
    async def shutdown(self) -> None: ...
    async def reserve_or_wait(
        self, size: int, *, net_memory_delta: int
    ) -> MemoryReservation: ...
    async def reserve_or_wait_or_overbook(
        self, size: int, *, net_memory_delta: int
    ) -> tuple[MemoryReservation, int]: ...
    async def reserve_or_wait_or_fail(
        self, size: int, *, net_memory_delta: int
    ) -> MemoryReservation: ...
    def size(self) -> int: ...

async def reserve_memory(
    ctx: Context,
    size: int,
    *,
    net_memory_delta: int,
    mem_type: MemoryType = MemoryType.DEVICE,
    allow_overbooking: bool | None = None,
) -> MemoryReservation: ...
