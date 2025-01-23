# Copyright (c) 2025, NVIDIA CORPORATION.
from __future__ import annotations

import pytest

import rmm.mr

from rapidsmp.buffer.resource import LimitAvailableMemory


def KiB(x: int) -> int:
    return x * 2**10


def test_limit_available_memory():
    with pytest.raises(
        TypeError,
        match="expected rmm.pylibrmm.memory_resource.StatisticsResourceAdaptor",
    ):
        LimitAvailableMemory(rmm.mr.CudaMemoryResource(), limit=KiB(100))

    mr = rmm.mr.StatisticsResourceAdaptor(rmm.mr.CudaMemoryResource())
    mem_available = LimitAvailableMemory(mr, limit=KiB(100))
    assert mem_available() == KiB(100)

    # Allocate a buffer reduces available memory.
    buf1 = rmm.DeviceBuffer(size=KiB(50), mr=mr)
    assert mem_available() == KiB(50)

    # But not when allocating using another memory resource.
    mr2 = rmm.mr.CudaMemoryResource()
    buf2 = rmm.DeviceBuffer(size=KiB(50), mr=mr2)
    assert mem_available() == KiB(50)
    del buf2

    # Available memory can be negative.
    buf3 = rmm.DeviceBuffer(size=KiB(100), mr=mr)
    assert mem_available() == -KiB(50)

    # Freeing buffers increases available memory.
    del buf1
    assert mem_available() == 0
    del buf3
    assert mem_available() == KiB(100)
