# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pylibcudf.table import Table
from rmm.pylibrmm.stream import Stream

from rapidsmpf.buffer.buffer import MemoryType

class TableChunk:
    @staticmethod
    def from_pylibcudf_table(
        sequence_number: int,
        table: Table,
        stream: Stream,
    ) -> TableChunk: ...
    def sequence_number(self) -> int: ...
    def stream(self) -> Stream: ...
    def data_alloc_size(self, mem_type: MemoryType) -> int: ...
    def is_available(self) -> bool: ...
    def table_view(self) -> Table: ...
