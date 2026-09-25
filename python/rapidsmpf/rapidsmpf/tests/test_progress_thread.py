# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.statistics import Statistics


def test_can_access_statistics_default() -> None:
    p = ProgressThread()
    assert not p.statistics.enabled


def test_can_access_statistics_provided() -> None:
    p = ProgressThread(Statistics(enable=True))
    assert p.statistics.enabled


def test_transfer_event_recorder_lifecycle() -> None:
    p = ProgressThread()
    p.enable_transfer_events(capacity=2)
    assert p.dropped_transfer_events == 0
    assert p.drain_transfer_events() == []
    assert p.drain_transfer_events() == []
    p.disable_transfer_events()

    with pytest.raises(RuntimeError, match="capacity must be positive"):
        p.enable_transfer_events(capacity=0)
