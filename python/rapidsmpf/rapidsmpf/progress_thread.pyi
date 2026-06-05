# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from rapidsmpf.runtime import Runtime

class ProgressThread:
    def __init__(self, runtime: Runtime) -> None: ...
