# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import pytest

import rmm
import rmm.mr

from rapidsmpf.memory.scoped_memory_record import AllocType, ScopedMemoryRecord
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream

KIB = 1024


def test_fallback_and_current_allocated() -> None:
    base = rmm.mr.CudaMemoryResource()
    state = {"main": 0, "fallback": 0}  # track which mr is used.

    def alloc_cb(
        size: int, stream: Stream, *, label: str, limit: int, track: list[int]
    ) -> int:
        if size > limit:
            raise MemoryError()
        ptr: int = base.allocate(size, stream)
        state[label] += size
        track.append(ptr)
        return ptr

    def dealloc_cb(
        ptr: int, size: int, stream: Stream, *, label: str, track: list[int]
    ) -> None:
        base.deallocate(ptr, size, stream)
        state[label] -= size
        track.append(ptr)

    main_track: list[int] = []
    fallback_track: list[int] = []

    main_mr = rmm.mr.CallbackMemoryResource(
        functools.partial(alloc_cb, label="main", limit=200 * KIB, track=main_track),
        functools.partial(dealloc_cb, label="main", track=main_track),
    )
    fallback_mr = rmm.mr.CallbackMemoryResource(
        functools.partial(
            alloc_cb, label="fallback", limit=1024 * KIB, track=fallback_track
        ),
        functools.partial(dealloc_cb, label="fallback", track=fallback_track),
    )
    mr_adaptor = RmmResourceAdaptor(upstream_mr=main_mr, fallback_mr=fallback_mr)

    # Delete upstream to check that adaptor keeps them alive.
    del main_mr
    del fallback_mr

    # Allocate buffer within main_mr limit.
    buf1 = rmm.DeviceBuffer(size=100 * KIB, mr=mr_adaptor)
    assert mr_adaptor.current_allocated == 100 * KIB
    assert state["main"] == 100 * KIB
    assert state["fallback"] == 0
    assert len(main_track) == 1
    assert len(fallback_track) == 0

    # Deallocate buffer.
    del buf1
    assert mr_adaptor.current_allocated == 0
    assert state["main"] == 0
    assert state["fallback"] == 0
    assert len(main_track) == 2  # alloc + dealloc

    # Allocate buffer too big for main_mr, should fall back.
    buf2 = rmm.DeviceBuffer(size=500 * KIB, mr=mr_adaptor)
    assert mr_adaptor.current_allocated == 500 * KIB
    assert state["main"] == 0
    assert state["fallback"] == 500 * KIB
    assert len(fallback_track) == 1

    # Deallocate.
    del buf2
    assert mr_adaptor.current_allocated == 0
    assert state["main"] == 0
    assert state["fallback"] == 0
    assert len(fallback_track) == 2  # alloc + dealloc


def test_except_type() -> None:
    def alloc_cb(size: int, stream: Stream) -> int:
        raise RuntimeError("not a MemoryError")

    def dealloc_cb(ptr: int, size: int, stream: Stream) -> None:
        return None

    mr = RmmResourceAdaptor(
        upstream_mr=rmm.mr.CallbackMemoryResource(alloc_cb, dealloc_cb),
        fallback_mr=rmm.mr.CudaMemoryResource(),
    )

    with pytest.raises(RuntimeError, match="not a MemoryError"):
        mr.allocate(1024)


@pytest.mark.parametrize(
    "alloc_type", [AllocType.PRIMARY, AllocType.FALLBACK, AllocType.ALL]
)
def test_initial_state(alloc_type: AllocType) -> None:
    record = ScopedMemoryRecord()
    assert record.num_total_allocs(alloc_type) == 0
    assert record.num_current_allocs(alloc_type) == 0
    assert record.current(alloc_type) == 0
    assert record.total(alloc_type) == 0
    assert record.peak(alloc_type) == 0


def test_single_allocation_and_deallocation() -> None:
    record = ScopedMemoryRecord()

    record.record_allocation(AllocType.PRIMARY, 1024)
    assert record.num_total_allocs(AllocType.PRIMARY) == 1
    assert record.num_current_allocs(AllocType.PRIMARY) == 1
    assert record.current(AllocType.PRIMARY) == 1024
    assert record.total(AllocType.PRIMARY) == 1024
    assert record.peak(AllocType.PRIMARY) == 1024

    record.record_deallocation(AllocType.PRIMARY, 1024)
    assert record.num_total_allocs(AllocType.PRIMARY) == 1  # total doesn't decrease
    assert record.num_current_allocs(AllocType.PRIMARY) == 0
    assert record.current(AllocType.PRIMARY) == 0
    assert record.total(AllocType.PRIMARY) == 1024
    assert record.peak(AllocType.PRIMARY) == 1024  # peak should stay


def test_multiple_allocators() -> None:
    record = ScopedMemoryRecord()

    record.record_allocation(AllocType.PRIMARY, 100)
    record.record_allocation(AllocType.FALLBACK, 300)

    assert record.num_total_allocs(AllocType.PRIMARY) == 1
    assert record.num_total_allocs(AllocType.FALLBACK) == 1
    assert record.num_total_allocs(AllocType.ALL) == 2

    assert record.num_current_allocs(AllocType.ALL) == 2
    assert record.current(AllocType.ALL) == 400
    assert record.total(AllocType.ALL) == 400
    assert record.peak(AllocType.ALL) == 400


def test_peak_tracking() -> None:
    record = ScopedMemoryRecord()

    record.record_allocation(AllocType.PRIMARY, 512)
    record.record_deallocation(AllocType.PRIMARY, 512)
    record.record_allocation(AllocType.PRIMARY, 256)

    assert record.peak(AllocType.PRIMARY) == 512
    assert record.current(AllocType.PRIMARY) == 256


def test_partial_deallocation() -> None:
    record = ScopedMemoryRecord()

    record.record_allocation(AllocType.FALLBACK, 800)
    record.record_deallocation(AllocType.FALLBACK, 300)
    assert record.num_current_allocs(AllocType.FALLBACK) == 0

    assert record.current(AllocType.FALLBACK) == 500
    # Allocation count unchanged by byte tracking
    assert record.num_total_allocs(AllocType.FALLBACK) == 1


def test_zero_allocation_behavior() -> None:
    record = ScopedMemoryRecord()

    record.record_allocation(AllocType.PRIMARY, 0)
    record.record_deallocation(AllocType.PRIMARY, 0)

    assert record.num_total_allocs(AllocType.PRIMARY) == 1
    assert record.current(AllocType.PRIMARY) == 0
    assert record.total(AllocType.PRIMARY) == 0
    assert record.peak(AllocType.PRIMARY) == 0
