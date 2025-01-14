# Copyright (c) 2025, NVIDIA CORPORATION.
from __future__ import annotations

from collections.abc import Iterable

from pylibcudf.contiguous_split import PackedColumns
from pylibcudf.table import Table

def partition_and_pack(
    table: Table, columns_to_hash: Iterable[int], num_partitions: int
) -> dict[int, PackedColumns]: ...
def unpack_and_concat(partitions: Iterable[PackedColumns]) -> Table: ...
