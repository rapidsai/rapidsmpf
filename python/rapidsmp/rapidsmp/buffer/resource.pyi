# Copyright (c) 2025, NVIDIA CORPORATION.

from collections.abc import Callable, Mapping

from rmm.pylibrmm.memory_resource import DeviceMemoryResource, StatisticsResourceAdaptor

from rapidsmp.buffer.buffer import MemoryType

class BufferResource:
    def __init__(
        self,
        device_mr: DeviceMemoryResource,
        memory_available: Mapping[MemoryType, Callable[[], int]] | None = None,
    ) -> None: ...
    def memory_reserved(self, mem_type: MemoryType) -> int: ...

class LimitAvailableMemory:
    def __init__(
        self,
        statistics_mr: StatisticsResourceAdaptor,
    ) -> None: ...
    def __call__(self) -> int: ...
