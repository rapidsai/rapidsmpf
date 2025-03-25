# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

class SpillManager:
    def add_spill_function(self, func: Callable[[int], int], priority: int) -> int: ...
    def spill(self, amount: int) -> int: ...
