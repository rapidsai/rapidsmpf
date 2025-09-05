# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import rmm.mr

from rapidsmpf.buffer.buffer import MemoryType
from rapidsmpf.buffer.resource import BufferResource, LimitAvailableMemory
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor


def KiB(x: int) -> int:
    return x * 2**10


def test_limit_available_memory() -> None:
    with pytest.raises(
        TypeError,
        match="RmmResourceAdaptor",
    ):
        LimitAvailableMemory(rmm.mr.CudaMemoryResource(), limit=KiB(100))

    mr = RmmResourceAdaptor(rmm.mr.CudaMemoryResource())
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


def test_buffer_resource() -> None:
    mr = RmmResourceAdaptor(rmm.mr.CudaMemoryResource())

    with pytest.raises(
        NotImplementedError,
        match="only accept `LimitAvailableMemory` as memory available functions",
    ):
        BufferResource(mr, {MemoryType.DEVICE: lambda: 42})

    mem_available = LimitAvailableMemory(mr, limit=KiB(100))
    br = BufferResource(mr, {MemoryType.DEVICE: mem_available})
    assert br.memory_reserved(MemoryType.DEVICE) == 0
    assert br.memory_reserved(MemoryType.HOST) == 0

    # Chack BufferResource.memory_available and BufferResource.device_memory_available
    assert br.memory_available(MemoryType.DEVICE) == KiB(100)
    assert br.device_memory_available == mem_available() == KiB(100)
    buf1 = rmm.DeviceBuffer(size=KiB(50), mr=mr)
    assert br.memory_available(MemoryType.DEVICE) == mem_available() == KiB(50)
    assert br.device_memory_available == mem_available() == KiB(50)
    del buf1
    assert br.memory_available(MemoryType.DEVICE) == mem_available() == KiB(100)
    assert br.device_memory_available == mem_available() == KiB(100)

    # TODO: add more `BufferResource` checks here as we add python bindings.
