# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Self

from rmm.pylibrmm.stream import Stream

from rapidsmpf.communicator.communicator import Logger
from rapidsmpf.config import Options
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.runtime import Runtime
from rapidsmpf.statistics import Statistics
from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.memory_reserve_or_wait import MemoryReserveOrWait
from rapidsmpf.streaming.core.message import PayloadT
from rapidsmpf.streaming.core.spillable_messages import SpillableMessages

class Context:
    def __init__(
        self,
        runtime: Runtime,
        br: BufferResource,
    ) -> None: ...
    @classmethod
    def from_options(
        cls: type[Self],
        runtime: Runtime,
        mr: RmmResourceAdaptor,
    ) -> Self: ...
    def __enter__(self) -> Context: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ) -> bool: ...
    def shutdown(self) -> None: ...
    def runtime(self) -> Runtime: ...
    def options(self) -> Options: ...
    def logger(self) -> Logger: ...
    def br(self) -> BufferResource: ...
    def statistics(self) -> Statistics: ...
    def get_stream_from_pool(self) -> Stream: ...
    def stream_pool_size(self) -> int: ...
    def create_channel(self) -> Channel[PayloadT]: ...
    def spillable_messages(self) -> SpillableMessages: ...
    def memory(self, mem_type: MemoryType) -> MemoryReserveOrWait: ...
