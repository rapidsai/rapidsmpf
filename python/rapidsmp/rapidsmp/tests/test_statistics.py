# Copyright (c) 2025, NVIDIA CORPORATION.
from __future__ import annotations

from rapidsmp.statistics import Statistics


def test_disabled():
    stats = Statistics(comm=None)
    assert not stats.enabled
