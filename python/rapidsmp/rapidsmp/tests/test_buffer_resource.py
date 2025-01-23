# Copyright (c) 2025, NVIDIA CORPORATION.
from __future__ import annotations

import pytest

import rmm.mr

from rapidsmp.buffer.resource import LimitAvailableMemory


def test_limit_available_memory():
    with pytest.raises(
        TypeError,
        match="expected rmm.pylibrmm.memory_resource.StatisticsResourceAdaptor",
    ):
        LimitAvailableMemory(rmm.mr.CudaMemoryResource(), limit=100)

    mr = rmm.mr.StatisticsResourceAdaptor(rmm.mr.CudaMemoryResource())
    LimitAvailableMemory(mr, limit=100)
