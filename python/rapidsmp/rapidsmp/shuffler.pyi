# Copyright (c) 2025, NVIDIA CORPORATION.
from __future__ import annotations

from collections.abc import Iterable, Mapping

from pylibcudf.contiguous_split import PackedColumns
from pylibcudf.table import Table
from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from rapidsmp.buffer.resource import BufferResource
from rapidsmp.communicator.communicator import Communicator
from rapidsmp.statistics import Statistics

def partition_and_pack(
    table: Table,
    columns_to_hash: Iterable[int],
    num_partitions: int,
    stream: Stream,
    device_mr: DeviceMemoryResource,
) -> dict[int, PackedColumns]: ...
def unpack_and_concat(
    partitions: Iterable[PackedColumns],
    stream: Stream,
    device_mr: DeviceMemoryResource,
) -> Table: ...

class Shuffler:
    def __init__(
        self,
        comm: Communicator,
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
    def insert_chunks(self, chunks: Mapping[int, PackedColumns]) -> None: ...
    def insert_finished(self, pid: int) -> None: ...
    def extract(self, pid: int) -> list[PackedColumns]: ...
    def finished(self) -> bool: ...
    def wait_any(self) -> int: ...
    def wait_on(self, pid: int) -> None: ...
