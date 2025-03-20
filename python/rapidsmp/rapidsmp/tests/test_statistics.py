# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from rapidsmp.statistics import Statistics


def test_disabled() -> None:
    stats = Statistics(enable=False)
    assert not stats.enabled
