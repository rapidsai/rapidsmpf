# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Self

from rapidsmpf.streaming.core.channel import Payload
from rapidsmpf.streaming.core.message import Message

class PartitionMapChunk:
    @classmethod
    def from_message(cls, message: Message[Self]) -> Self: ...
    def into_message(self, message: Message[Self]) -> None: ...
    @property
    def sequence_number(self) -> int: ...

class PartitionVectorChunk:
    @classmethod
    def from_message(cls, message: Message[Self]) -> Self: ...
    def into_message(self, message: Message[Self]) -> None: ...
    @property
    def sequence_number(self) -> int: ...

if TYPE_CHECKING:
    # Check that PartitionMapChunk implements Payload.
    t1: PartitionMapChunk
    t2: Payload = t1

    # Check that PartitionVectorChunk implements Payload.
    t3: PartitionVectorChunk
    t4: Payload = t3
