# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.statistics import Statistics

if TYPE_CHECKING:
    import rmm.mr


def test_disabled() -> None:
    stats = Statistics(enable=False)
    assert not stats.enabled


def test_add_get_stat() -> None:
    stats = Statistics(enable=True)
    assert stats.enabled

    stats.add_stat("stat1", 1)
    stats.add_stat("stat2", 2)
    stats.add_stat("stat1", 4)
    assert stats.get_stat("stat1") == {"count": 2, "value": 5.0}
    assert stats.get_stat("stat2") == {"count": 1, "value": 2}


def test_get_nonexistent_stat() -> None:
    """Test that accessing a non-existent statistic raises KeyError."""
    stats = Statistics(enable=True)

    with pytest.raises(KeyError, match="Statistic 'foo' does not exist"):
        stats.get_stat("foo")


def test_enable_memory_profiling(device_mr: rmm.mr.CudaMemoryResource) -> None:
    stats = Statistics(enable=False)
    assert not stats.memory_profiling_enabled
    mr = RmmResourceAdaptor(device_mr)
    stats = Statistics(enable=False, mr=mr)
    assert not stats.memory_profiling_enabled
    stats = Statistics(enable=True, mr=mr)
    assert stats.memory_profiling_enabled


def test_get_empty_memory_records(device_mr: rmm.mr.CudaMemoryResource) -> None:
    stats = Statistics(enable=True, mr=RmmResourceAdaptor(device_mr))
    assert stats.get_memory_records() == {}


def test_memory_profiling(device_mr: rmm.mr.CudaMemoryResource) -> None:
    mr = RmmResourceAdaptor(device_mr)
    stats = Statistics(enable=True, mr=mr)
    with stats.memory_profiling("outer"):
        b1 = mr.allocate(1024)
        with stats.memory_profiling("inner"):
            mr.deallocate(mr.allocate(512), 512)
            mr.deallocate(mr.allocate(512), 512)
        mr.deallocate(b1, 1024)
        mr.deallocate(mr.allocate(1024), 1024)

    inner = stats.get_memory_records()["inner"]
    assert inner.scoped.num_total_allocs() == 2
    assert inner.scoped.peak() == 512
    assert inner.scoped.total() == 512 + 512
    assert inner.global_peak == 512
    assert inner.num_calls == 1

    outer = stats.get_memory_records()["outer"]
    assert outer.scoped.num_total_allocs() == 2 + 2
    assert outer.scoped.peak() == 1024 + 512
    assert outer.scoped.total() == 1024 + 1024 + 512 + 512
    assert outer.global_peak == 1024 + 512
    assert outer.num_calls == 1


def test_list_stat_names() -> None:
    stats = Statistics(enable=True)
    assert stats.list_stat_names() == []
    stats.add_stat("stat1", 1.0)
    assert stats.list_stat_names() == ["stat1"]
    stats.add_stat("stat2", 2.0)
    assert stats.list_stat_names() == ["stat1", "stat2"]
    stats.add_stat("stat1", 3.0)
    assert stats.list_stat_names() == ["stat1", "stat2"]


def test_clear() -> None:
    # stats
    stats = Statistics(enable=True)
    stats.add_stat("stat1", 10.0)

    assert stats.get_stat("stat1") == {"count": 1, "value": 10.0}
    stats.clear()
    assert stats.list_stat_names() == []

    stats.add_stat("stat1", 10.0)
    assert stats.get_stat("stat1") == {"count": 1, "value": 10.0}
