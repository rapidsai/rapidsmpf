# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Self

from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.streaming.core.message import Message

class PartitionMapChunk:
    @classmethod
    def from_message(
        cls: type[Self], message: Message[Self], br: BufferResource
    ) -> Self: ...
    def into_message(self, sequence_number: int, message: Message[Self]) -> None: ...

class PartitionVectorChunk:
    @classmethod
    def from_message(
        cls: type[Self], message: Message[Self], br: BufferResource
    ) -> Self: ...
    def into_message(self, sequence_number: int, message: Message[Self]) -> None: ...
