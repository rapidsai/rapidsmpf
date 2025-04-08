# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Mapping

from rmm.pylibrmm.memory_resource import DeviceMemoryResource, StatisticsResourceAdaptor

from rapidsmp.buffer.buffer import MemoryType
from rapidsmp.buffer.spill_manager import SpillManager

class BufferResource:
    def __init__(
        self,
        device_mr: DeviceMemoryResource,
        memory_available: Mapping[MemoryType, Callable[[], int]] | None = None,
        periodic_spill_check: float | None = ...,
    ) -> None: ...
    def memory_reserved(self, mem_type: MemoryType) -> int: ...
    @property
    def spill_manager(self) -> SpillManager: ...

class LimitAvailableMemory:
    def __init__(
        self,
        statistics_mr: StatisticsResourceAdaptor,
        limit: int,
    ) -> None: ...
    def __call__(self) -> int: ...
