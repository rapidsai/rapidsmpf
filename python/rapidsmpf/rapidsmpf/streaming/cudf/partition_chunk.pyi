# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Self

from pylibcudf.table import Table
from rmm.pylibrmm.stream import Stream

from rapidsmpf.streaming.core.channel import Message, Payload

class PartitionMapChunk:
    @staticmethod
    def from_pylibcudf_table(
        sequence_number: int,
        table: Table,
        stream: Stream,
    ) -> PartitionMapChunk: ...
    @classmethod
    def from_message(cls, message: Message[Self]) -> Self: ...
    def into_message(self, message: Message[Self]) -> None: ...
    def sequence_number(self) -> int: ...
    def stream(self) -> Stream: ...

if TYPE_CHECKING:
    # Check that PartitionMapChunk implements Payload.
    tc: PartitionMapChunk
    p: Payload = tc
