# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable, Mapping

from rmm.pylibrmm.stream import Stream

from rapidsmpf.buffer.packed_data import PackedData
from rapidsmpf.buffer.resource import BufferResource
from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.statistics import Statistics

class Shuffler:
    max_concurrent_shuffles: int
    def __init__(
        self,
        comm: Communicator,
        progress_thread: ProgressThread,
        op_id: int,
        total_num_partitions: int,
        stream: Stream,
        br: BufferResource,
        statistics: Statistics | None = None,
    ) -> None: ...
    def shutdown(self) -> None: ...
    def __str__(self) -> str: ...
    @property
    def comm(self) -> Communicator: ...
    def insert_chunks(self, chunks: Mapping[int, PackedData]) -> None: ...
    def concat_insert(self, chunks: Mapping[int, PackedData]) -> None: ...
    def insert_finished(self, pids: int | Iterable[int]) -> None: ...
    def extract(self, pid: int) -> list[PackedData]: ...
    def finished(self) -> bool: ...
    def wait_any(self) -> int: ...
    def wait_on(self, pid: int) -> None: ...
