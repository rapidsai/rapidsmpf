# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Self

from rmm.pylibrmm.stream import Stream

from rapidsmpf.streaming.core.channel import Message, Payload

class PartitionMapChunk:
    @classmethod
    def from_message(cls, message: Message[Self]) -> Self: ...
    def into_message(self, message: Message[Self]) -> None: ...
    @property
    def sequence_number(self) -> int: ...
    @property
    def stream(self) -> Stream: ...

class PartitionVectorChunk:
    @classmethod
    def from_message(cls, message: Message[Self]) -> Self: ...
    def into_message(self, message: Message[Self]) -> None: ...
    @property
    def sequence_number(self) -> int: ...
    @property
    def stream(self) -> Stream: ...

if TYPE_CHECKING:
    # Check that PartitionMapChunk implements Payload.
    t1: PartitionMapChunk
    t2: Payload = t1

    # Check that PartitionVectorChunk implements Payload.
    t3: PartitionVectorChunk
    t4: Payload = t3
