# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Generic

from rapidsmpf.streaming.core.channel import Channel, Message, PayloadT
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.node import CppNode

class DeferredMessages(Generic[PayloadT]):
    def release(self) -> list[Message[PayloadT]]: ...

def push_to_channel(
    ctx: Context, ch_out: Channel[PayloadT], messages: list[Message[PayloadT]]
) -> CppNode: ...
def pull_from_channel(
    ctx: Context, ch_in: Channel[PayloadT]
) -> tuple[CppNode, DeferredMessages[PayloadT]]: ...
