# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Submodule for streaming core operations."""

from __future__ import annotations

from rapidsmpf.streaming.core.actor import define_actor, run_actor_network
from rapidsmpf.streaming.core.cancellation import shutdown_channels
from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.fanout import FanoutPolicy, fanout
from rapidsmpf.streaming.core.memory_reserve_or_wait import (
    MemoryReserveOrWait,
    reserve_memory,
)
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.spillable_messages import SpillableMessages

__all__ = [
    "Channel",
    "Context",
    "FanoutPolicy",
    "MemoryReserveOrWait",
    "Message",
    "SpillableMessages",
    "define_actor",
    "fanout",
    "reserve_memory",
    "run_actor_network",
    "shutdown_channels",
]
