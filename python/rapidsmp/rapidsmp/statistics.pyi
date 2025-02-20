# Copyright (c) 2025, NVIDIA CORPORATION.
from __future__ import annotations

class Statistics:
    def __init__(
        self,
        nranks: int,
    ) -> None: ...
    @property
    def enabled(self) -> bool: ...
    def report(self) -> str: ...
