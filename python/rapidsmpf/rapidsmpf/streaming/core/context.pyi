# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from rapidsmpf.buffer.resource import BufferResource
from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.config import Options
from rapidsmpf.statistics import Statistics

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
