# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Self, overload

from pylibcudf.table import Table
from rmm.pylibrmm.stream import Stream

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.memory_reservation import MemoryReservation
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.message import Message, Payload

class TableChunk:
    @staticmethod
    def from_pylibcudf_table(
        table: Table,
        stream: Stream,
        *,
        exclusive_view: bool,
    ) -> TableChunk: ...
    @classmethod
    def from_message(cls: type[Self], message: Message[Self]) -> Self: ...
    def into_message(self, sequence_number: int, message: Message[Self]) -> None: ...
    @property
    def stream(self) -> Stream: ...
    def data_alloc_size(self, mem_type: MemoryType | None = None) -> int: ...
    def is_available(self) -> bool: ...
    def make_available_cost(self) -> int: ...
    def make_available(self, reservation: MemoryReservation) -> TableChunk: ...
    async def make_available_or_wait(
        self, ctx: Context, *, net_memory_delta: int
    ) -> TableChunk: ...
    def make_available_and_spill(
        self, br: BufferResource, *, allow_overbooking: bool
    ) -> TableChunk: ...
    def table_view(self) -> Table: ...
    def is_spillable(self) -> bool: ...
    def copy(self, reservation: MemoryReservation) -> TableChunk: ...

@overload
async def make_table_chunks_available_or_wait(
    context: Context,
    chunks: TableChunk,
    *,
    reserve_extra: int,
    net_memory_delta: int,
    allow_overbooking: bool | None = None,
) -> tuple[TableChunk, MemoryReservation]: ...
@overload
async def make_table_chunks_available_or_wait(
    context: Context,
    chunks: Iterable[TableChunk],
    *,
    reserve_extra: int,
    net_memory_delta: int,
    allow_overbooking: bool | None = None,
) -> tuple[list[TableChunk], MemoryReservation]: ...

if TYPE_CHECKING:
    # Check that TableChunk implements Payload.
    tc: TableChunk
    p: Payload = tc
