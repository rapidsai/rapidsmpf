# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Generic

from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.message import PayloadT

class Lineariser(Generic[PayloadT]):
    def __init__(self, output: Channel[PayloadT], num_producers: int) -> None: ...
    def get_inputs(self) -> list[Channel[PayloadT]]: ...
    async def drain(self, ctx: Context) -> None: ...
