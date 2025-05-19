# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

class RmmFallbackResource:
    def __init__(
        self,
        upstream_mr: DeviceMemoryResource,
        alternate_upstream_mr: DeviceMemoryResource,
    ): ...
    @property
    def get_upstream(self) -> DeviceMemoryResource: ...
    @property
    def get_alternate_upstream(self) -> DeviceMemoryResource: ...
