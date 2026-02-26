# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from os import PathLike
from typing import Any, Self

from rapidsmpf.config import Options
from rapidsmpf.memory.scoped_memory_record import ScopedMemoryRecord
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor

class Statistics:
    def __init__(
        self,
        *,
        enable: bool,
        mr: RmmResourceAdaptor | None = None,
    ) -> None: ...
    @classmethod
    def from_options(
        cls: type[Self], mr: RmmResourceAdaptor, options: Options
    ) -> Self: ...
    @property
    def enabled(self) -> bool: ...
    def report(self) -> str: ...
    def get_stat(self, name: str) -> dict[str, Number]: ...
    def list_stat_names(self) -> list[str]: ...
    def add_stat(self, name: str, value: float) -> None: ...
    @property
    def memory_profiling_enabled(self) -> bool: ...
    def get_memory_records(self) -> dict[str, MemoryRecord]: ...
    def memory_profiling(self, name: str) -> MemoryRecorder: ...
    def clear(self) -> None: ...
    def write_json(self, filepath: str | PathLike[str]) -> None: ...
    def write_json_string(self) -> str: ...

@dataclass
class MemoryRecord:
    scoped: ScopedMemoryRecord
    global_peak: int
    num_calls: int

class MemoryRecorder:
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ) -> bool: ...
