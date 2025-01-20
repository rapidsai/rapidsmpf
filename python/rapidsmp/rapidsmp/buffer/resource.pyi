# Copyright (c) 2025, NVIDIA CORPORATION.

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

class BufferResource:
    def __init__(
        self,
        device_mr: DeviceMemoryResource,
    ) -> None: ...
