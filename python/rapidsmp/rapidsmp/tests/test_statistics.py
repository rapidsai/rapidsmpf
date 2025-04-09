# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from rapidsmp.statistics import Statistics


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
