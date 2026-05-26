# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import rmm
import rmm.mr

from rapidsmpf.memory.scoped_memory_record import ScopedMemoryRecord
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor

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
    mr_adaptor = RmmResourceAdaptor(upstream_mr=upstream_mr)

    # Delete upstream to check that adaptor keeps it alive.
    del upstream_mr

    buf = rmm.DeviceBuffer(size=100 * KIB, mr=mr_adaptor)
    assert mr_adaptor.current_allocated == 100 * KIB
    assert state["bytes"] == 100 * KIB
    assert len(track) == 1

    del buf
    assert mr_adaptor.current_allocated == 0
    assert state["bytes"] == 0
    assert len(track) == 2  # alloc + dealloc


def test_except_type() -> None:
    def alloc_cb(size: int, stream: Stream) -> int:
        raise RuntimeError("not a MemoryError")

    def dealloc_cb(ptr: int, size: int, stream: Stream) -> None:
        return None

    mr = RmmResourceAdaptor(
        upstream_mr=rmm.mr.CallbackMemoryResource(alloc_cb, dealloc_cb),
    )

    with pytest.raises(RuntimeError, match="not a MemoryError"):
        mr.allocate(1024)


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
