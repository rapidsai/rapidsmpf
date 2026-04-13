# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Self

from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.core.message import Message

class PackedDataChunk:
    def to_packed_data(self) -> PackedData: ...
    @staticmethod
    def from_packed_data(obj: PackedData, br: BufferResource) -> PackedDataChunk: ...
    @classmethod
    def from_message(
        cls: type[Self], message: Message[Self], br: BufferResource
    ) -> Self: ...
    def into_message(self, sequence_number: int, message: Message[Self]) -> None: ...
