# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import rmm
import rmm.mr

from rapidsmpf.buffer.rmm_fallback_resource import RmmFallbackResource

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream


def test_fallback_resource_adaptor() -> None:
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
    alternate_track: list[int] = []
    alternate_mr = rmm.mr.CallbackMemoryResource(
        functools.partial(alloc_cb, track=alternate_track, limit=1000),
        functools.partial(dealloc_cb, track=alternate_track),
    )
    mr = RmmFallbackResource(main_mr, alternate_mr)
    assert main_mr is mr.get_upstream()
    assert alternate_mr is mr.get_alternate_upstream()

    # Delete the upstream memory resources here to check that they are
    # kept alive by `mr`
    del main_mr
    del alternate_mr

    # Buffer size within the limit of `main_mr`
    rmm.DeviceBuffer(size=100, mr=mr)
    # we expect an alloc and a dealloc of the same buffer in
    # `main_track` and an empty `alternate_track`
    assert len(main_track) == 2
    assert main_track[0] == main_track[1]
    assert len(alternate_track) == 0

    # Buffer size outside the limit of `main_mr`
    rmm.DeviceBuffer(size=500, mr=mr)
    # we expect an alloc and a dealloc of the same buffer in
    # `alternate_track` and an unchanged `main_mr`
    assert len(main_track) == 2
    assert main_track[0] == main_track[1]
    assert len(alternate_track) == 2
    assert alternate_track[0] == alternate_track[1]
