# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context


def test_context(context: Context) -> None:
    print(context)
