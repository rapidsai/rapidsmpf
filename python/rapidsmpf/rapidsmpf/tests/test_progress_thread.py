# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.statistics import Statistics


def test_can_access_statistics_default() -> None:
    p = ProgressThread()
    assert not p.statistics.enabled


def test_can_access_statistics_provided() -> None:
    p = ProgressThread(Statistics(enable=True))
    assert p.statistics.enabled
