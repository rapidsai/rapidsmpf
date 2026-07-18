# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from typing import Self

from rmm.pylibrmm.stream import Stream

from rapidsmpf.config import Options

def is_pinned_memory_resources_supported() -> bool: ...
@dataclass
class PinnedPoolProperties:
    initial_pool_size: int = 0
    max_pool_size: int | None = None
    numa_id: int | None = None

class PinnedMemoryResource:
    @property
    def enabled(self) -> bool: ...
    def allocate(self, nbytes: int, stream: Stream) -> int: ...
    def deallocate(self, ptr: int, nbytes: int, stream: Stream) -> None: ...
