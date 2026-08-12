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
from rapidsmpf.memory.pinned_memory_resource import (
    PinnedPoolProperties,
    is_pinned_memory_resources_supported,
)

skip_if_no_pinned = pytest.mark.skipif(
    not is_pinned_memory_resources_supported(),
    reason="pinned memory pool not supported on this system",
)


@pytest.fixture
def pinned_br() -> BufferResource:
    pool_size = 4 * 1024 * 1024
    mr = rmm.mr.CudaMemoryResource()
    return BufferResource(
        mr,
        pinned_pool_properties=PinnedPoolProperties(initial_pool_size=pool_size),
        memory_limits={MemoryType.PINNED_HOST: pool_size},
    )


@pytest.mark.parametrize(
    "mem_type",
    [
        MemoryType.DEVICE,
        MemoryType.HOST,
        pytest.param(MemoryType.PINNED_HOST, marks=skip_if_no_pinned),
    ],
)
def test_make_buffer(mem_type: MemoryType) -> None:
    size = 1024
    stream = Stream()
    if mem_type == MemoryType.PINNED_HOST:
        mr = rmm.mr.CudaMemoryResource()
        pool_size = 4 * 1024 * 1024
        br = BufferResource(
            mr,
            pinned_pool_properties=PinnedPoolProperties(initial_pool_size=pool_size),
            memory_limits={MemoryType.PINNED_HOST: pool_size},
        )
    else:
        mr = rmm.mr.CudaMemoryResource()
        br = BufferResource(mr, memory_limits={mem_type: size * 4})
    reservation, _ = br.reserve(mem_type, size, allow_overbooking=False)
    buf = br.make_buffer(size, stream, reservation)
    assert isinstance(buf, Buffer)
    assert buf.size == size
    assert buf.mem_type == mem_type


@skip_if_no_pinned
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


def test_buffer_protocol_host() -> None:
    size = 256
    mr = rmm.mr.CudaMemoryResource()
    br = BufferResource(mr, memory_limits={MemoryType.HOST: size * 4})
    stream = Stream()
    reservation, _ = br.reserve(MemoryType.HOST, size, allow_overbooking=False)
    buf = br.make_buffer(size, stream, reservation)

    mv = memoryview(buf)
    assert len(mv) == size
    assert mv.format == "B"


def test_buffer_protocol_rejects_device() -> None:
    size = 1024
    mr = rmm.mr.CudaMemoryResource()
    br = BufferResource(mr, memory_limits={MemoryType.DEVICE: size * 4})
    stream = Stream()
    reservation, _ = br.reserve(MemoryType.DEVICE, size, allow_overbooking=False)
    buf = br.make_buffer(size, stream, reservation)

    with pytest.raises(TypeError, match="host buffers"):
        memoryview(buf)
