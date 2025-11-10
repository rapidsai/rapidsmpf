# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from typing import Generic

from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.message import Message, PayloadT

class Channel(Generic[PayloadT]):
    @abstractmethod  # Mark it abstract to force the use of `Context.create_channel()`.
    def __init__(self) -> None: ...
    async def drain(self, ctx: Context) -> None: ...
    async def shutdown(self, ctx: Context) -> None: ...
    async def send(self, ctx: Context, item: Message[PayloadT]) -> None: ...
    async def recv(self, ctx: Context) -> Message[PayloadT] | None: ...
