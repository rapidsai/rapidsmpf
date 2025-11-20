# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Self

from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.core.message import Message, Payload

class PackedDataChunk:
    def to_packed_data(self) -> PackedData: ...
    @staticmethod
    def from_packed_data(obj: PackedData) -> PackedDataChunk: ...
    @classmethod
    def from_message(cls: type[Self], message: Message[Self]) -> Self: ...
    def into_message(self, sequence_number: int, message: Message[Self]) -> None: ...

if TYPE_CHECKING:
    t1: PackedDataChunk
    t2: Payload = t1
