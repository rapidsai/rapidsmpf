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

    # Check BufferResource.memory_available
    assert br.memory_available(MemoryType.DEVICE) == mem_available() == KiB(100)
    buf1 = rmm.DeviceBuffer(size=KiB(50), mr=mr)
    assert br.memory_available(MemoryType.DEVICE) == mem_available() == KiB(50)
    del buf1
    assert br.memory_available(MemoryType.DEVICE) == mem_available() == KiB(100)


@pytest.mark.parametrize("mem_type", [MemoryType.DEVICE, MemoryType.HOST])
def test_memory_reservation(mem_type: MemoryType) -> None:
    mr = RmmResourceAdaptor(rmm.mr.CudaMemoryResource())
    br = BufferResource(mr, {mem_type: LimitAvailableMemory(mr, limit=KiB(100))})
    res1, ob = br.reserve(mem_type, KiB(100), allow_overbooking=False)
    assert res1.br is br
    assert res1.mem_type == mem_type
    assert res1.size == KiB(100)
    assert ob == 0

    # Allow overbooking.
    res2, ob = br.reserve(mem_type, KiB(100), allow_overbooking=True)
    assert res2.br is br
    assert res2.mem_type == mem_type
    assert res2.size == KiB(100)
    assert ob == KiB(100)

    # Disallow overbooking, but reserve() never fails. Instead, the size
    # of the reservation becomes zero.
    res3, ob = br.reserve(mem_type, KiB(100), allow_overbooking=False)
    assert res3.br is br
    assert res3.mem_type == mem_type
    assert res3.size == KiB(0)
    assert ob == KiB(200)

    # Deleting reservations, lowers the current overbooking.
    del res3
    _, ob = br.reserve(mem_type, KiB(0), allow_overbooking=False)
    assert ob == KiB(100)
    del res2
    _, ob = br.reserve(mem_type, KiB(0), allow_overbooking=False)
    assert ob == KiB(0)

    # We can also partial release a reservation.
    assert br.release(res1, KiB(60)) == KiB(40)
    assert res1.size == KiB(40)
    assert br.release(res1, KiB(40)) == KiB(0)
    assert res1.size == KiB(0)

    # But a reservation cannot go to negative.
    with pytest.raises(
        OverflowError,
        match="isn't big enough",
    ):
        br.release(res1, KiB(10))


def test_stream_pool() -> None:
    """Test that stream_pool parameter can be configured."""
    mr = RmmResourceAdaptor(rmm.mr.CudaMemoryResource())

    # Test with default stream pool size (16)
    br_default = BufferResource(mr)
    assert br_default.stream_pool_size() == 16

    # Test with custom stream pool
    custom_pool = rmm.pylibrmm.cuda_stream_pool.CudaStreamPool(pool_size=32)
    br_custom = BufferResource(mr, stream_pool=custom_pool)
    assert br_custom.stream_pool_size() == 32
