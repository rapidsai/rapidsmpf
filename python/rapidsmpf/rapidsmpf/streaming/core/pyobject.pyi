# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from rapidsmpf.streaming.core.channel import Message, Payload

class PyObjectPayload:
    @staticmethod
    def from_object(sequence_number: int, obj: Any) -> PyObjectPayload: ...
    @classmethod
    def from_message(cls, message: Message[Self]) -> Self: ...
    def into_message(self, message: Message[Self]) -> None: ...
    @property
    def sequence_number(self) -> int: ...
    def extract_object(self) -> Any: ...

if TYPE_CHECKING:
    # Check that PyObjectPayload implements Payload.
    _t1: PyObjectPayload
    _t2: Payload = _t1
