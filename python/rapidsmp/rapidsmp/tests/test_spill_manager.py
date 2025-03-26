# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rapidsmp.buffer.resource import BufferResource

if TYPE_CHECKING:
    import rmm.mr


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

    br = BufferResource(device_mr)
    br.spill_manager.add_spill_function(spill, 0)
    with pytest.raises(error):
        br.spill_manager.spill(10)


def test_spill_function(
    device_mr: rmm.mr.CudaMemoryResource,
) -> None:
    br = BufferResource(device_mr)
    track_spilled = [0]

    def spill_unlimited(amount: int) -> int:
        track_spilled[0] += amount
        return amount

    f1 = br.spill_manager.add_spill_function(spill_unlimited, priority=0)
    assert br.spill_manager.spill(10) == 10
    assert track_spilled[0] == 10

    def spill_not_needed(amount: int) -> int:
        raise ValueError("shouldn't be needed")

    f2 = br.spill_manager.add_spill_function(spill_not_needed, priority=-1)
    assert br.spill_manager.spill(10) == 10
    assert track_spilled[0] == 20

    def spill_limited(amount: int) -> int:
        return 5

    f3 = br.spill_manager.add_spill_function(spill_limited, priority=1)
    assert br.spill_manager.spill(10) == 10
    assert track_spilled[0] == 25

    br.spill_manager.remove_spill_function(f3)
    assert br.spill_manager.spill(10) == 10
    assert track_spilled[0] == 35

    br.spill_manager.remove_spill_function(f1)
    with pytest.raises(ValueError, match="shouldn't be needed"):
        br.spill_manager.spill(10)

    br.spill_manager.remove_spill_function(f2)
    assert br.spill_manager.spill(10) == 0
    assert track_spilled[0] == 35
