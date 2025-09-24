# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from rmm.pylibrmm.stream import Stream

from rapidsmpf.buffer.packed_data import PackedData
from rapidsmpf.buffer.resource import BufferResource
from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.statistics import Statistics

class AllGather:
    def __init__(
        self,
        comm: Communicator,
        progress_thread: ProgressThread,
        op_id: int,
        stream: Stream,
        br: BufferResource,
        statistics: Statistics | None = None,
    ) -> None: ...
    @property
    def comm(self) -> Communicator: ...
    def insert(self, packed_data: PackedData) -> None: ...
    def insert_finished(self) -> None: ...
    def finished(self) -> bool: ...
    def wait_and_extract(
        self, ordered: bool = True, timeout_ms: int = -1
    ) -> list[PackedData]: ...
    def extract_ready(self) -> list[PackedData]: ...
