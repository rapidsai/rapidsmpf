# Copyright (c) 2025, NVIDIA CORPORATION.

from rmm.pylibrmm.memory_resource import DeviceMemoryResource, StatisticsResourceAdaptor

class BufferResource:
    def __init__(
        self,
        device_mr: DeviceMemoryResource,
    ) -> None: ...

class LimitAvailableMemory:
    def __init__(
        self,
        statistics_mr: StatisticsResourceAdaptor,
    ) -> None: ...
