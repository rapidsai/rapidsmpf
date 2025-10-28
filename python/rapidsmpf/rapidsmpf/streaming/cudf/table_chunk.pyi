# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Self

from pylibcudf.table import Table
from rmm.pylibrmm.stream import Stream

from rapidsmpf.buffer.buffer import MemoryType
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
    def from_message(cls, message: Message[Self]) -> Self: ...
    def into_message(self, sequence_number: int, message: Message[Self]) -> None: ...
    @property
    def stream(self) -> Stream: ...
    def data_alloc_size(self, mem_type: MemoryType) -> int: ...
    def is_available(self) -> bool: ...
    def table_view(self) -> Table: ...
    def is_spillable(self) -> bool: ...

if TYPE_CHECKING:
    # Check that TableChunk implements Payload.
    tc: TableChunk
    p: Payload = tc
