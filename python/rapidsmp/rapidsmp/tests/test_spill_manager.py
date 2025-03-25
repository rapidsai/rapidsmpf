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
def test_error_handle(
    device_mr: rmm.mr.CudaMemoryResource, error: type[Exception]
) -> None:
    def spill(amount: int) -> int:
        raise error

    br = BufferResource(device_mr)
    br.spill_manager.add_spill_function(spill, 0)
    with pytest.raises(error):
        br.spill_manager.spill(10)
