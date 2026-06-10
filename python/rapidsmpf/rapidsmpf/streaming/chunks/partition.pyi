# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Self

from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.core.message import Message

class PartitionMapChunk:
    @classmethod
    def from_packed_data_map(
        cls: type[Self], data: Mapping[int, PackedData], br: BufferResource
    ) -> Self: ...
    @classmethod
    def from_message(
        cls: type[Self], message: Message[Self], br: BufferResource
    ) -> Self: ...
    def to_packed_data_map(self) -> dict[int, PackedData]: ...
    def into_message(self, sequence_number: int, message: Message[Self]) -> None: ...

class PartitionVectorChunk:
    @classmethod
    def from_packed_data_list(
        cls: type[Self], data: Sequence[PackedData], br: BufferResource
    ) -> Self: ...
    @classmethod
    def from_message(
        cls: type[Self], message: Message[Self], br: BufferResource
    ) -> Self: ...
    def to_packed_data_list(self) -> list[PackedData]: ...
    def into_message(self, sequence_number: int, message: Message[Self]) -> None: ...
