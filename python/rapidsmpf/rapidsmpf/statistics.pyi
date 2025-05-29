# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from numbers import Number

from rmm.pylibrmm.memory_resource import StatisticsResourceAdaptor

class Statistics:
    def __init__(
        self,
        enable: bool,
        mr: StatisticsResourceAdaptor | None = None,
    ) -> None: ...
    @property
    def enabled(self) -> bool: ...
    def report(self) -> str: ...
    def get_stat(self, name: str) -> dict[str, Number]: ...
    def add_stat(self, name: str, value: float) -> float: ...
