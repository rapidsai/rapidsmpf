# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

import rmm
import rmm.mr
from rmm.pylibrmm.stream import Stream

from rapidsmpf.memory.buffer import Buffer, MemoryType
from rapidsmpf.memory.buffer_resource import BufferResource


@pytest.fixture
def pinned_br() -> BufferResource:
    from rapidsmpf.memory.pinned_memory_resource import PinnedPoolProperties

    pool_size = 4 * 1024 * 1024
    mr = rmm.mr.CudaMemoryResource()
    return BufferResource(
        mr,
        pinned_pool_properties=PinnedPoolProperties(initial_pool_size=pool_size),
        memory_limits={MemoryType.PINNED_HOST: pool_size},
    )


def test_make_buffer_pinned_host(pinned_br: BufferResource) -> None:
    size = 1024
    stream = Stream()
    reservation, _ = pinned_br.reserve(
        MemoryType.PINNED_HOST, size, allow_overbooking=False
    )
    buf = pinned_br.make_buffer(size, stream, reservation)
    assert isinstance(buf, Buffer)
    assert buf.size == size
    assert buf.mem_type == MemoryType.PINNED_HOST


def test_make_buffer_device() -> None:
    size = 1024
    mr = rmm.mr.CudaMemoryResource()
    br = BufferResource(mr, memory_limits={MemoryType.DEVICE: size * 4})
    stream = Stream()
    reservation, _ = br.reserve(MemoryType.DEVICE, size, allow_overbooking=False)
    buf = br.make_buffer(size, stream, reservation)
    assert isinstance(buf, Buffer)
    assert buf.size == size
    assert buf.mem_type == MemoryType.DEVICE


def test_buffer_protocol_pinned_host(pinned_br: BufferResource) -> None:
    size = 256
    stream = Stream()
    reservation, _ = pinned_br.reserve(
        MemoryType.PINNED_HOST, size, allow_overbooking=False
    )
    buf = pinned_br.make_buffer(size, stream, reservation)

    mv = memoryview(buf)
    assert len(mv) == size
    assert mv.format == "B"

    data = np.arange(size, dtype=np.uint8)
    np.frombuffer(mv, dtype=np.uint8)[:] = data
    assert np.array_equal(np.frombuffer(memoryview(buf), dtype=np.uint8), data)


def test_buffer_protocol_requires_pinned_host() -> None:
    size = 1024
    mr = rmm.mr.CudaMemoryResource()
    br = BufferResource(mr, memory_limits={MemoryType.DEVICE: size * 4})
    stream = Stream()
    reservation, _ = br.reserve(MemoryType.DEVICE, size, allow_overbooking=False)
    buf = br.make_buffer(size, stream, reservation)

    with pytest.raises(TypeError, match="PINNED_HOST"):
        memoryview(buf)
