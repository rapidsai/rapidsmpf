# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import pytest

import rmm
import rmm.mr

from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream


def test_fallback() -> None:
    base = rmm.mr.CudaMemoryResource()

    def alloc_cb(size: int, stream: Stream, *, track: list[int], limit: int) -> Any:
        if size > limit:
            raise MemoryError()
        ret = base.allocate(size, stream)
        track.append(ret)
        return ret

    def dealloc_cb(ptr: int, size: int, stream: Stream, *, track: list[int]) -> Any:
        track.append(ptr)
        return base.deallocate(ptr, size, stream)

    main_track: list[int] = []
    main_mr = rmm.mr.CallbackMemoryResource(
        functools.partial(alloc_cb, track=main_track, limit=200),
        functools.partial(dealloc_cb, track=main_track),
    )
    fallback_track: list[int] = []
    fallback_mr = rmm.mr.CallbackMemoryResource(
        functools.partial(alloc_cb, track=fallback_track, limit=1000),
        functools.partial(dealloc_cb, track=fallback_track),
    )
    mr = RmmResourceAdaptor(upstream_mr=main_mr, fallback_mr=fallback_mr)

    # Delete the upstream memory resources here to check that they are
    # kept alive by `mr`.
    del main_mr
    del fallback_mr

    # Buffer size within the limit of `main_mr`.
    rmm.DeviceBuffer(size=100, mr=mr)
    # we expect an alloc and a dealloc of the same buffer in
    # `main_track` and an empty `fallback_track`.
    assert len(main_track) == 2
    assert main_track[0] == main_track[1]
    assert len(fallback_track) == 0

    # Buffer size outside the limit of `main_mr`
    rmm.DeviceBuffer(size=500, mr=mr)
    # we expect an alloc and a dealloc of the same buffer in
    # `fallback_track` and an unchanged `main_mr`.
    assert len(main_track) == 2
    assert main_track[0] == main_track[1]
    assert len(fallback_track) == 2
    assert fallback_track[0] == fallback_track[1]


def test_except_type() -> None:
    def alloc_cb(size: int, stream: Stream) -> Any:
        raise RuntimeError("not a MemoryError")

    def dealloc_cb(ptr: int, size: int, stream: Stream) -> Any:
        return None

    mr = RmmResourceAdaptor(
        upstream_mr=rmm.mr.CallbackMemoryResource(alloc_cb, dealloc_cb),
        fallback_mr=rmm.mr.CudaMemoryResource(),
    )

    with pytest.raises(RuntimeError, match="not a MemoryError"):
        mr.allocate(100)
