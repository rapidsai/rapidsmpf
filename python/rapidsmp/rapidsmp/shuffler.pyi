# Copyright (c) 2025, NVIDIA CORPORATION.
from __future__ import annotations

from collections.abc import Iterable, Mapping

from pylibcudf.contiguous_split import PackedColumns
from pylibcudf.table import Table
from rmm._cuda.stream import Stream

from rapidsmp.buffer.resource import BufferResource
from rapidsmp.communicator.communicator import Communicator

def partition_and_pack(
    table: Table,
    columns_to_hash: Iterable[int],
    num_partitions: int,
    stream: Stream,
) -> dict[int, PackedColumns]: ...
def unpack_and_concat(partitions: Iterable[PackedColumns]) -> Table: ...

class Shuffler:
    def __init__(
        self,
        comm: Communicator,
        total_num_partitions: int,
        stream: Stream,
        br: BufferResource,
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
