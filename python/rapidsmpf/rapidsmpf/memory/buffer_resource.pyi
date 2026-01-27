# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Mapping
from typing import Self

from rmm.pylibrmm.cuda_stream_pool import CudaStreamPool
from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from rapidsmpf.config import Options
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.memory_reservation import MemoryReservation
from rapidsmpf.memory.pinned_memory_resource import PinnedMemoryResource
from rapidsmpf.memory.spill_manager import SpillManager
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.statistics import Statistics

class BufferResource:
    def __init__(
        self,
        device_mr: DeviceMemoryResource,
        *,
        pinned_mr: PinnedMemoryResource | None = None,
        memory_available: Mapping[MemoryType, Callable[[], int]]
        | AvailableMemoryMap
        | None = None,
        periodic_spill_check: float | None = 1e-3,
        stream_pool: CudaStreamPool | None = None,
        statistics: Statistics | None = None,
    ) -> None: ...
    @property
    def device_mr(self) -> DeviceMemoryResource: ...
    @property
    def pinned_mr(self) -> PinnedMemoryResource | None: ...
    def memory_available(self, mem_type: MemoryType) -> int: ...
    def memory_reserved(self, mem_type: MemoryType) -> int: ...
    @property
    def spill_manager(self) -> SpillManager: ...
    @property
    def statistics(self) -> Statistics: ...
    def reserve(
        self, mem_type: MemoryType, size: int, *, allow_overbooking: bool
    ) -> tuple[MemoryReservation, int]: ...
    def reserve_device_memory_and_spill(
        self, size: int, *, allow_overbooking: bool
    ) -> MemoryReservation: ...
    def release(self, reservation: MemoryReservation, size: int) -> int: ...
    def stream_pool_size(self) -> int: ...

class LimitAvailableMemory:
    def __init__(
        self,
        mr: RmmResourceAdaptor,
        limit: int,
    ) -> None: ...
    def __call__(self) -> int: ...

class AvailableMemoryMap:
    @classmethod
    def from_options(
        cls: type[Self], mr: RmmResourceAdaptor, options: Options
    ) -> Self: ...

def periodic_spill_check_from_options(options: Options) -> float | None: ...
def stream_pool_from_options(options: Options) -> int: ...
