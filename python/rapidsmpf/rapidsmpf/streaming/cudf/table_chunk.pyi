# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Self

from pylibcudf.table import Table
from rmm.pylibrmm.stream import Stream

from rapidsmpf.buffer.buffer import MemoryType
from rapidsmpf.buffer.resource import BufferResource, MemoryReservation
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
    def data_alloc_size(self, mem_type: MemoryType) -> int: ...
    def is_available(self) -> bool: ...
    def make_available_cost(self) -> int: ...
    def make_available(self, reservation: MemoryReservation) -> TableChunk: ...
    def make_available_and_spill(
        self, br: BufferResource, *, allow_overbooking: bool
    ) -> TableChunk: ...
    def table_view(self) -> Table: ...
    def is_spillable(self) -> bool: ...
    def copy(self, reservation: MemoryReservation) -> TableChunk: ...

if TYPE_CHECKING:
    # Check that TableChunk implements Payload.
    tc: TableChunk
    p: Payload = tc
