# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import IntEnum

from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.node import CppNode

class FanoutPolicy(IntEnum):
    BOUNDED = ...
    UNBOUNDED = ...

def fanout(
    ctx: Context,
    ch_in: Channel,
    chs_out: list[Channel],
    policy: FanoutPolicy,
) -> CppNode: ...
