# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable, Mapping

from pylibcudf.table import Table
from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from rapidsmpf.buffer.packed_data import PackedData
from rapidsmpf.buffer.resource import BufferResource
from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.statistics import Statistics

def partition_and_pack(
    table: Table,
    columns_to_hash: Iterable[int],
    num_partitions: int,
    stream: Stream,
    device_mr: DeviceMemoryResource,
) -> dict[int, PackedData]: ...
def split_and_pack(
    table: Table,
    splits: Iterable[int],
    stream: Stream,
    device_mr: DeviceMemoryResource,
) -> dict[int, PackedData]: ...
def unpack_and_concat(
    partitions: Iterable[PackedData],
    stream: Stream,
    device_mr: DeviceMemoryResource,
) -> Table: ...

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
    def insert_finished(self, pid: int) -> None: ...
    def extract(self, pid: int) -> list[PackedData]: ...
    def finished(self) -> bool: ...
    def wait_any(self) -> int: ...
    def wait_on(self, pid: int) -> None: ...
