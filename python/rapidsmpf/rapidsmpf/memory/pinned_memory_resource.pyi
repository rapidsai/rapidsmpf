# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from typing import Self

from rapidsmpf.config import Options

def is_pinned_memory_resources_supported() -> bool: ...

class PinnedMemoryResource:
    def __init__(self, numa_id: int | None = None): ...
    @staticmethod
    def make_if_available(
        numa_id: int | None = None,
    ) -> PinnedMemoryResource | None: ...
    @classmethod
    def from_options(cls: type[Self], options: Options) -> Self: ...
