# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection, Generator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Concatenate, ParamSpec

from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context

P = ParamSpec("P")

class CppActor:
    pass

class PyActor(Awaitable[None]):
    def __await__(self) -> Generator[Any, None, None]: ...

def define_actor(
    *, extra_channels: Collection[Channel] = ()
) -> Callable[
    [Callable[Concatenate[Context, P], Awaitable[None]]],
    Callable[Concatenate[Context, P], PyActor],
]: ...
def run_actor_graph(
    *,
    nodes: Collection[CppActor | PyActor],
    py_executor: ThreadPoolExecutor | None = None,
) -> None: ...
