# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection, Generator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Concatenate, ParamSpec

from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context

P = ParamSpec("P")

class CppNode:
    pass

class PyNode(Awaitable[None]):
    def __await__(self) -> Generator[Any, None, None]: ...

def define_py_node(
    *, extra_channels: Collection[Channel] = ()
) -> Callable[
    [Callable[Concatenate[Context, P], Awaitable[None]]],
    Callable[Concatenate[Context, P], PyNode],
]: ...
def run_streaming_pipeline(
    *,
    nodes: Collection[CppNode | PyNode],
    py_executor: ThreadPoolExecutor | None = None,
) -> None: ...
