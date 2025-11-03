# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Generic, Protocol, TypeVar

PayloadT = TypeVar("PayloadT", bound="Payload")

class Payload(Protocol):
    """
    Protocol for the payload of a Message.

    Any object sent through a Channel must implement this protocol.
    It defines how to reconstruct the payload from a message and how to
    insert it back into a message.

    Methods
    -------
    from_message(message)
        Construct a payload instance by consuming a message.
    into_message(sequence_number, message)
        Insert the payload into a message. The payload instance is released
        in the process.
    """

    @classmethod
    def from_message(cls: PayloadT, message: Message[PayloadT]) -> PayloadT: ...
    def into_message(
        self: PayloadT, sequence_number: int, message: Message[PayloadT]
    ) -> None: ...

class Message(Generic[PayloadT]):
    def __init__(self, sequence_number: int, payload: PayloadT): ...
    def empty(self) -> bool: ...
    @property
    def sequence_number(self) -> int: ...
