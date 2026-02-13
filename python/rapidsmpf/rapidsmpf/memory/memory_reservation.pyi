# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import contextmanager

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.buffer_resource import BufferResource

class MemoryReservation:
    @property
    def size(self) -> int: ...
    @property
    def mem_type(self) -> MemoryType: ...
    @property
    def br(self) -> BufferResource: ...
    def clear(self) -> None: ...

@contextmanager
def opaque_memory_usage(
    reservation: MemoryReservation,
) -> Iterator[MemoryReservation]: ...
