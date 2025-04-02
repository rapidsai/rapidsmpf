# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from rapidsmp.buffer.buffer import MemoryType
from rapidsmp.buffer.spill_collection import SpillCollection


class MySpillableObject:
    def __init__(self, nbytes: int) -> None:
        self._mem_type = MemoryType.DEVICE
        self._nbytes = nbytes

    def mem_type(self) -> MemoryType:
        return self._mem_type

    def approx_spillable_amount(self) -> int:
        return self._nbytes

    def spill(self, amount: int) -> int:
        self._mem_type = MemoryType.HOST
        return self._nbytes


def test_spill_collection() -> None:
    collection = SpillCollection()
    obj = MySpillableObject(100)

    collection.add_spillable(obj)
    assert collection.spill(100) == 100
    assert collection.spill(100) == 0
