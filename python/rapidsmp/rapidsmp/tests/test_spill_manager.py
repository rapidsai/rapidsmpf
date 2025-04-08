# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time

import pytest

import rmm.mr

from rapidsmp.buffer.buffer import MemoryType
from rapidsmp.buffer.resource import BufferResource, LimitAvailableMemory


@pytest.mark.parametrize(
    "error",
    [
        MemoryError,
        TypeError,
        ValueError,
        IOError,
        IndexError,
        OverflowError,
        ArithmeticError,
        RuntimeError,
    ],
)
def test_error_handling(
    device_mr: rmm.mr.CudaMemoryResource, error: type[Exception]
) -> None:
    def spill(amount: int) -> int:
        raise error

    br = BufferResource(device_mr, periodic_spill_check=None)
    br.spill_manager.add_spill_function(spill, 0)
    with pytest.raises(error):
        br.spill_manager.spill(10)


def test_spill_function(
    device_mr: rmm.mr.CudaMemoryResource,
) -> None:
    br = BufferResource(device_mr, periodic_spill_check=None)
    track_spilled = [0]

    def spill_unlimited(amount: int) -> int:
        track_spilled[0] += amount
        return amount

    f1 = br.spill_manager.add_spill_function(spill_unlimited, priority=0)
    assert br.spill_manager.spill(10) == 10
    assert track_spilled[0] == 10
    track_spilled[0] = 0

    def spill_not_needed(amount: int) -> int:
        raise ValueError("shouldn't be needed")

    f2 = br.spill_manager.add_spill_function(spill_not_needed, priority=-1)
    assert br.spill_manager.spill(10) == 10
    assert track_spilled[0] == 10
    track_spilled[0] = 0

    def spill_limited(amount: int) -> int:
        return 5

    f3 = br.spill_manager.add_spill_function(spill_limited, priority=1)
    assert br.spill_manager.spill(10) == 10
    assert track_spilled[0] == 5  # Note `track_spilled` doesn't track `spill_limited`.
    track_spilled[0] = 0

    br.spill_manager.remove_spill_function(f3)
    assert br.spill_manager.spill(10) == 10
    assert track_spilled[0] == 10

    br.spill_manager.remove_spill_function(f1)
    with pytest.raises(ValueError, match="shouldn't be needed"):
        br.spill_manager.spill(10)

    br.spill_manager.remove_spill_function(f2)
    assert br.spill_manager.spill(10) == 0
    assert track_spilled[0] == 10


def test_spill_function_outlive_buffer_resource(
    device_mr: rmm.mr.CudaMemoryResource,
) -> None:
    spill_manager = BufferResource(device_mr, periodic_spill_check=None).spill_manager
    with pytest.raises(ValueError):
        spill_manager.add_spill_function(lambda x: x, 0)
    with pytest.raises(ValueError):
        spill_manager.remove_spill_function(0)
    with pytest.raises(ValueError):
        spill_manager.spill(10)


def test_periodic_spill_check(
    device_mr: rmm.mr.CudaMemoryResource,
) -> None:
    # Create a buffer resource with a negative limit to trigger spilling and
    # a periodic spill check enabled.
    mr = rmm.mr.StatisticsResourceAdaptor(device_mr)
    mem_available = LimitAvailableMemory(mr, limit=-100)
    br = BufferResource(
        mr,
        memory_available={MemoryType.DEVICE: mem_available},
        periodic_spill_check=0.001,
    )

    track_spilled = [0]

    def spill(amount: int) -> int:
        track_spilled[0] += amount
        return amount

    f1 = br.spill_manager.add_spill_function(spill, priority=0)
    # After a short sleep, we expect many calls to `spill()` by the periodic check.
    time.sleep(0.1)
    assert track_spilled[0] > 0
    del f1
