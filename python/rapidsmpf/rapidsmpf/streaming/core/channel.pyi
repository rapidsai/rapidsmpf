# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Generic, Protocol, Self, TypeVar

from rapidsmpf.streaming.core.context import Context

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
    into_message(message)
        Insert the payload into a message. The payload instance is released
        in the process.
    """

    @classmethod
    def from_message(cls, message: Message[Self]) -> Self: ...
    def into_message(self, message: Message[Self]) -> None: ...

class Message(Generic[PayloadT]):
    def __init__(self, payload: PayloadT): ...
    def empty(self) -> bool: ...

class Channel(Message[PayloadT]):
    def __init__(self) -> None: ...
    async def drain(self, ctx: Context) -> None: ...
    async def shutdown(self, ctx: Context) -> None: ...
    async def send(self, ctx: Context, item: Message[PayloadT]) -> None: ...
    async def recv(self, ctx: Context) -> Message[PayloadT] | None: ...
