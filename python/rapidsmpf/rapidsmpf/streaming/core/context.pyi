# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from rmm.pylibrmm.stream import Stream

from rapidsmpf.buffer.resource import BufferResource
from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.config import Options
from rapidsmpf.statistics import Statistics
from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.message import PayloadT

class Context:
    def __init__(
        self,
        comm: Communicator,
        br: BufferResource,
        options: Options | None = None,
        statistics: Statistics | None = None,
    ) -> None: ...
    def options(self) -> Options: ...
    def comm(self) -> Communicator: ...
    def br(self) -> BufferResource: ...
    def statistics(self) -> Statistics: ...
    def get_stream_from_pool(self) -> Stream: ...
    def stream_pool_size(self) -> int: ...
    def create_channel(self) -> Channel[PayloadT]: ...
