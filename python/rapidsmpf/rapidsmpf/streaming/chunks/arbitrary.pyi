# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar

from rapidsmpf.streaming.core.message import Message, Payload

T = TypeVar("T")

class ArbitraryChunk(Generic[T]):
    def __init__(self, obj: T) -> None: ...
    def release(self) -> T: ...
    @classmethod
    def from_message(cls: type[Self], message: Message[Self]) -> Self: ...
    def into_message(self, sequence_number: int, message: Message[Self]) -> None: ...

if TYPE_CHECKING:
    # Check that ArbitraryChunk implements Payload.
    t1: ArbitraryChunk[Any]
    t2: Payload = t1
