# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

from rapidsmpf.memory.buffer import MemoryType

class SpillManager:
    def add_spill_function(
        self, func: Callable[[MemoryType, int], int], priority: int
    ) -> int: ...
    def remove_spill_function(self, function_id: int) -> None: ...
    def spill(self, amount: int, mem_type: MemoryType | None = None) -> int: ...
