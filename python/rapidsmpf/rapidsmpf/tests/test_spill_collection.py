# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.spill_collection import SpillCollection, Spillable

if TYPE_CHECKING:
    import rmm
    from rmm.pylibrmm.memory_resource import DeviceMemoryResource
    from rmm.pylibrmm.stream import Stream


class MySpillableObject:
    def __init__(self, nbytes: int) -> None:
        self._mem_type = MemoryType.DEVICE
        self._nbytes = nbytes

    def mem_type(self) -> MemoryType:
        return self._mem_type

    def approx_spillable_amount(self) -> int:
        return self._nbytes

    def spill(
        self,
        amount: int,
        *,
        stream: Stream,
        device_mr: DeviceMemoryResource,
        staging_device_buffer: rmm.DeviceBuffer | None = None,
    ) -> int:
        self._mem_type = MemoryType.HOST
        return self._nbytes

    def unspill(self) -> MySpillableObject:
        self._mem_type = MemoryType.DEVICE
        return self


def test_spill_collection(stream: Stream, device_mr: DeviceMemoryResource) -> None:
    collection = SpillCollection()

    obj1 = MySpillableObject(100)
    assert isinstance(obj1, Spillable)
    collection.add_spillable(obj1)
    assert collection.spill(100, stream=stream, device_mr=device_mr) == 100
    assert collection.spill(100, stream=stream, device_mr=device_mr) == 0
    assert obj1.mem_type() == MemoryType.HOST

    obj2 = MySpillableObject(10)
    obj3 = MySpillableObject(10)
    collection.add_spillable(obj2)
    collection.add_spillable(obj3)
    # Eventhough we ask for 100 bytes, only 20 bytes can be spilled.
    assert collection.spill(100, stream=stream, device_mr=device_mr) == 20
    assert collection.spill(100, stream=stream, device_mr=device_mr) == 0
    assert obj2.mem_type() == MemoryType.HOST
    assert obj3.mem_type() == MemoryType.HOST

    # Check that we can unspill and re-spill an object.
    obj1 = obj1.unspill()
    assert obj1.mem_type() == MemoryType.DEVICE
    assert collection.spill(100, stream=stream, device_mr=device_mr) == 100
    assert collection.spill(100, stream=stream, device_mr=device_mr) == 0
    assert obj1.mem_type() == MemoryType.HOST
