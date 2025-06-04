# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable

from pylibcudf.table import Table
from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from rapidsmpf.buffer.packed_data import PackedData

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
