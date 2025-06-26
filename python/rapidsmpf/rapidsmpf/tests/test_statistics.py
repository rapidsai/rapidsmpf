# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

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

    assert stats.add_stat("stat1", 1) == 1
    assert stats.add_stat("stat2", 2) == 2
    assert stats.add_stat("stat1", 4) == 5
    assert stats.get_stat("stat1") == {"count": 2, "value": 5.0}
    assert stats.get_stat("stat2") == {"count": 1, "value": 2}


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
    stats = Statistics(enable=True, mr=RmmResourceAdaptor(device_mr))
    with stats.memory_profiling("test-scope"):
        pass
