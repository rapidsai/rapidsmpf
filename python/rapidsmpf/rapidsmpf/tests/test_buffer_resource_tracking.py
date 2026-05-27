# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import rmm
import rmm.mr

from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.scoped_memory_record import ScopedMemoryRecord

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream

KIB = 1024


def test_tracks_allocations() -> None:
    base = rmm.mr.CudaMemoryResource()
    state = {"bytes": 0}
    track: list[int] = []

    def alloc_cb(size: int, stream: Stream) -> int:
        ptr: int = base.allocate(size, stream)
        state["bytes"] += size
        track.append(ptr)
        return ptr

    def dealloc_cb(ptr: int, size: int, stream: Stream) -> None:
        base.deallocate(ptr, size, stream)
        state["bytes"] -= size
        track.append(ptr)

    upstream_mr = rmm.mr.CallbackMemoryResource(alloc_cb, dealloc_cb)
    br = BufferResource(upstream_mr)

    # Delete upstream to check that BR keeps it alive.
    del upstream_mr

    buf = rmm.DeviceBuffer(size=100 * KIB, mr=br)
    assert br.current_allocated == 100 * KIB
    assert state["bytes"] == 100 * KIB
    assert len(track) == 1

    del buf
    assert br.current_allocated == 0
    assert state["bytes"] == 0
    assert len(track) == 2  # alloc + dealloc


def test_except_type() -> None:
    def alloc_cb(size: int, stream: Stream) -> int:
        raise RuntimeError("not a MemoryError")

    def dealloc_cb(ptr: int, size: int, stream: Stream) -> None:
        return None

    br = BufferResource(rmm.mr.CallbackMemoryResource(alloc_cb, dealloc_cb))

    with pytest.raises(RuntimeError, match="not a MemoryError"):
        br.allocate(1024)


def test_lifetime_no_dangling_stream_pool() -> None:
    # rmm.DeviceBuffer keeps the BufferResource alive via the inherited
    # DeviceMemoryResource owning ref. Dropping the local handle to BR
    # must not invalidate the buffer's underlying stream pool.
    def make() -> rmm.DeviceBuffer:
        br = BufferResource(rmm.mr.CudaMemoryResource())
        return rmm.DeviceBuffer(size=1024, mr=br)

    buf = make()
    # Before the merge, this would dangle once the local `br` went out of
    # scope. Now the buffer keeps the BR (and its stream pool) alive.
    buf.copy_from_host(b"x" * 1024)


def test_initial_state() -> None:
    record = ScopedMemoryRecord()
    assert record.num_total_allocs() == 0
    assert record.num_current_allocs() == 0
    assert record.current() == 0
    assert record.total() == 0
    assert record.peak() == 0


def test_single_allocation_and_deallocation() -> None:
    record = ScopedMemoryRecord()

    record.record_allocation(1024)
    assert record.num_total_allocs() == 1
    assert record.num_current_allocs() == 1
    assert record.current() == 1024
    assert record.total() == 1024
    assert record.peak() == 1024

    record.record_deallocation(1024)
    assert record.num_total_allocs() == 1  # total doesn't decrease
    assert record.num_current_allocs() == 0
    assert record.current() == 0
    assert record.total() == 1024
    assert record.peak() == 1024  # peak should stay


def test_peak_tracking() -> None:
    record = ScopedMemoryRecord()

    record.record_allocation(512)
    record.record_deallocation(512)
    record.record_allocation(256)

    assert record.peak() == 512
    assert record.current() == 256


def test_partial_deallocation() -> None:
    record = ScopedMemoryRecord()

    record.record_allocation(800)
    record.record_deallocation(300)
    assert record.num_current_allocs() == 0

    assert record.current() == 500
    # Allocation count unchanged by byte tracking
    assert record.num_total_allocs() == 1


def test_zero_allocation_behavior() -> None:
    record = ScopedMemoryRecord()

    record.record_allocation(0)
    record.record_deallocation(0)

    assert record.num_total_allocs() == 1
    assert record.current() == 0
    assert record.total() == 0
    assert record.peak() == 0
