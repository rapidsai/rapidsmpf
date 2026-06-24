# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import rmm
import rmm.mr

from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.scoped_memory_record import ScopedMemoryRecord
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream

KIB = 1024


def test_cannot_construct_directly() -> None:
    """RmmResourceAdaptor is not constructible from Python.

    A usable, copyable adaptor is always owned by a ``BufferResource`` (which
    installs the back-reference). Direct construction must raise.
    """
    with pytest.raises(TypeError, match="device_mr_adaptor"):
        RmmResourceAdaptor(rmm.mr.CudaMemoryResource())
    with pytest.raises(TypeError, match="device_mr_adaptor"):
        RmmResourceAdaptor()


def test_tracks_allocations() -> None:
    """A BufferResource-backed adaptor tracks allocations made through it."""
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
    mr_adaptor = br.device_mr_adaptor()

    # Drop external references; the back-ref'd adaptor keeps the BufferResource
    # (and hence the upstream) alive. The adaptor is copyable, so it can be
    # stored in an owning container such as rmm.DeviceBuffer.
    del upstream_mr, br

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

    br = BufferResource(rmm.mr.CallbackMemoryResource(alloc_cb, dealloc_cb))
    mr = br.device_mr_adaptor()

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
