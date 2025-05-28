# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

class RmmResourceAdaptor:
    def __init__(
        self,
        upstream_mr: DeviceMemoryResource,
        *,
        fallback_mr: DeviceMemoryResource | None = None,
    ): ...
    @property
    def get_upstream(self) -> DeviceMemoryResource: ...
    def allocate(self, nbytes: int, stream: Stream = ...) -> int: ...
    def deallocate(self, ptr: int, nbytes: int, stream: Stream = ...) -> None: ...
    @property
    def current_allocated(self) -> int: ...
