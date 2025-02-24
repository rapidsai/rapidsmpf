# Copyright (c) 2025, NVIDIA CORPORATION.
from __future__ import annotations

from rapidsmp.communicator.communicator import Communicator

class Statistics:
    def __init__(
        self,
        comm: Communicator | None,
    ) -> None: ...
    @property
    def enabled(self) -> bool: ...
    def report(self) -> str: ...
