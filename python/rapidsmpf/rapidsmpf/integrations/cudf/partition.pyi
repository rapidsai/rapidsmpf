# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable

from pylibcudf.table import Table
from rmm.pylibrmm.stream import Stream

from rapidsmpf.buffer.packed_data import PackedData
from rapidsmpf.buffer.resource import BufferResource
from rapidsmpf.statistics import Statistics

def partition_and_pack(
    table: Table,
    columns_to_hash: Iterable[int],
    num_partitions: int,
    stream: Stream,
    br: BufferResource,
) -> dict[int, PackedData]: ...
def split_and_pack(
    table: Table,
    splits: Iterable[int],
    stream: Stream,
    br: BufferResource,
) -> dict[int, PackedData]: ...
def unpack_and_concat(
    partitions: Iterable[PackedData],
    stream: Stream,
    br: BufferResource,
) -> Table: ...
def spill_partitions(
    partitions: Iterable[PackedData],
    *,
    br: BufferResource,
    statistics: Statistics | None = None,
) -> list[PackedData]: ...
def unspill_partitions(
    partitions: Iterable[PackedData],
    *,
    br: BufferResource,
    allow_overbooking: bool,
    statistics: Statistics | None = None,
) -> list[PackedData]: ...
