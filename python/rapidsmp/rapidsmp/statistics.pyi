# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

class Statistics:
    def __init__(
        self,
        enable: bool,
    ) -> None: ...
    @property
    def enabled(self) -> bool: ...
    def report(self) -> str: ...
