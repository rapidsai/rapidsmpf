# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

cudf = pytest.importorskip("cudf")

from rapidsmpf.examples.streaming import basic_example  # noqa: E402


def test_basic_streaming_example() -> None:
    basic_example.main()
