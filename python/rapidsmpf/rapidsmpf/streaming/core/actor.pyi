# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection
from concurrent.futures import ThreadPoolExecutor
from typing import Concatenate, ParamSpec

from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context

P = ParamSpec("P")

class CppActor:
    pass

def define_actor(
    *, extra_channels: Collection[Channel] = ()
) -> Callable[
    [Callable[Concatenate[Context, P], Awaitable[None]]],
    Callable[Concatenate[Context, P], Awaitable[None]],
]: ...
def run_actor_network(
    *,
    actors: Collection[CppActor | Awaitable[None]],
    py_executor: ThreadPoolExecutor | None = None,
) -> None: ...
