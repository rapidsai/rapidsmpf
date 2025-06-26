# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from numbers import Number

from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor

class Statistics:
    def __init__(
        self,
        *,
        enable: bool,
        mr: RmmResourceAdaptor | None = None,
    ) -> None: ...
    @property
    def enabled(self) -> bool: ...
    def report(self) -> str: ...
    def get_stat(self, name: str) -> dict[str, Number]: ...
    def add_stat(self, name: str, value: float) -> float: ...
    @property
    def memory_profiling_enabled(self) -> bool: ...
