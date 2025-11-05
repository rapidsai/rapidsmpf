# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Mapping

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from rapidsmpf.buffer.buffer import MemoryType
from rapidsmpf.buffer.spill_manager import SpillManager
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor

class MemoryReservation:
    @property
    def size(self) -> int: ...
    @property
    def mem_type(self) -> MemoryType: ...
    @property
    def br(self) -> BufferResource: ...

class BufferResource:
    def __init__(
        self,
        device_mr: DeviceMemoryResource,
        memory_available: Mapping[MemoryType, Callable[[], int]] | None = None,
        periodic_spill_check: float | None = 1e-3,
    ) -> None: ...
    def memory_available(self, mem_type: MemoryType) -> int: ...
    def memory_reserved(self, mem_type: MemoryType) -> int: ...
    @property
    def spill_manager(self) -> SpillManager: ...
    def reserve(
        self, mem_type: MemoryType, size: int, *, allow_overbooking: bool
    ) -> tuple[MemoryReservation, int]: ...
    def release(self, reservation: MemoryReservation, size: int) -> int: ...

class LimitAvailableMemory:
    def __init__(
        self,
        mr: RmmResourceAdaptor,
        limit: int,
    ) -> None: ...
    def __call__(self) -> int: ...
