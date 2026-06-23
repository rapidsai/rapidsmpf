# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable

from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.packed_data import PackedData

def spill_partitions(
    partitions: Iterable[PackedData],
    *,
    br: BufferResource,
) -> list[PackedData]: ...
def unspill_partitions(
    partitions: Iterable[PackedData],
    *,
    br: BufferResource,
    allow_overbooking: bool,
) -> list[PackedData]: ...
