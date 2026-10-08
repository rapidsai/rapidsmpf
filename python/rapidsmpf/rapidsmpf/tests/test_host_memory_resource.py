# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rmm.pylibrmm.stream import Stream

from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.host_memory_resource import HostMemoryResource
from rapidsmpf.statistics import Statistics

if TYPE_CHECKING:
    import rmm.mr


def test_cannot_construct_directly() -> None:
    with pytest.raises(TypeError, match="cannot be constructed directly"):
        HostMemoryResource()


def test_allocate_deallocate_tracks_usage(device_mr: rmm.mr.CudaMemoryResource) -> None:
    host_mr = BufferResource(device_mr).host_mr
    stream = Stream()
    assert host_mr.current_allocated == 0

    ptr = host_mr.allocate(1024, stream)
    assert ptr != 0
    assert host_mr.current_allocated == 1024
    record = host_mr.get_main_memory_record()
    assert record.num_total_allocs() == 1
    assert record.current() == 1024

    host_mr.deallocate(ptr, 1024, stream)
    assert host_mr.current_allocated == 0
    assert host_mr.get_main_memory_record().peak() == 1024


def test_handle_keeps_buffer_resource_alive(
    device_mr: rmm.mr.CudaMemoryResource,
) -> None:
    host_mr = BufferResource(device_mr).host_mr
    stream = Stream()
    ptr = host_mr.allocate(256, stream)
    assert host_mr.current_allocated == 256
    host_mr.deallocate(ptr, 256, stream)


def test_report_includes_host_memory(device_mr: rmm.mr.CudaMemoryResource) -> None:
    br = BufferResource(device_mr)
    host_mr = br.host_mr
    stats = Statistics(enable=True)
    stream = Stream()
    ptr = host_mr.allocate(2048, stream)
    try:
        without = stats.report(mr=br.device_mr_adaptor())
        with_host = stats.report(mr=br.device_mr_adaptor(), host_mr=host_mr)
        with_header = stats.report(
            mr=br.device_mr_adaptor(), host_mr=host_mr, header="Run 1:"
        )
    finally:
        host_mr.deallocate(ptr, 2048, stream)

    assert "HostMemoryResource" not in without
    assert "HostMemoryResource" in with_host
    assert with_header.startswith("Run 1:")
    assert "HostMemoryResource" in with_header
