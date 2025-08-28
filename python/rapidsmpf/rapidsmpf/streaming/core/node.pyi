# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection, Generator
from concurrent.futures import ThreadPoolExecutor
from typing import ParamSpec

from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context

P = ParamSpec("P")

class CppNode:
    pass

class PyNode(Awaitable[None]):
    def __await__(self) -> Generator[object, None, None]: ...

def run_streaming_pipeline(
    *,
    nodes: Collection[CppNode | Awaitable[None]],
    py_executor: ThreadPoolExecutor | None = None,
) -> None: ...
def define_py_node(
    ctx: Context, *, channels: Collection[Channel] = ()
) -> Callable[[Callable[P, Awaitable[None]]], Callable[P, PyNode]]: ...
