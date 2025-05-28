# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import pytest

import rmm
import rmm.mr

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
