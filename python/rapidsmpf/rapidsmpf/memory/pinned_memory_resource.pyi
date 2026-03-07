# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from typing import Self

from rapidsmpf.config import Options

def is_pinned_memory_resources_supported() -> bool: ...

class PinnedPoolProperties:
    initial_pool_size: int
    max_pool_size: int | None
    def __init__(
        self, initial_pool_size: int = 0, max_pool_size: int | None = None
    ) -> None: ...

class PinnedMemoryResource:
    def __init__(
        self,
        numa_id: int | None = None,
        pool_properties: PinnedPoolProperties | None = None,
    ) -> None: ...
    @staticmethod
    def make_if_available(
        numa_id: int | None = None,
        pool_properties: PinnedPoolProperties | None = None,
    ) -> PinnedMemoryResource | None: ...
    @classmethod
    def from_options(cls: type[Self], options: Options) -> Self | None: ...
