# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from rmm.pylibrmm.stream import Stream

from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.config import Options
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.statistics import Statistics
from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.memory_reserve_or_wait import MemoryReserveOrWait
from rapidsmpf.streaming.core.message import PayloadT
from rapidsmpf.streaming.core.spillable_messages import SpillableMessages

class Context:
    def __init__(
        self,
        comm: Communicator,
        br: BufferResource,
        options: Options | None = None,
        statistics: Statistics | None = None,
    ) -> None: ...
    def __enter__(self) -> Context: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ) -> bool: ...
    def shutdown(self) -> None: ...
    def options(self) -> Options: ...
    def comm(self) -> Communicator: ...
    def br(self) -> BufferResource: ...
    def statistics(self) -> Statistics: ...
    def get_stream_from_pool(self) -> Stream: ...
    def stream_pool_size(self) -> int: ...
    def create_channel(self) -> Channel[PayloadT]: ...
    def spillable_messages(self) -> SpillableMessages: ...
    def memory(self, mem_type: MemoryType) -> MemoryReserveOrWait: ...
