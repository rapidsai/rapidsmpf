# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
from collections.abc import Mapping
from enum import IntEnum
from typing import Literal, Self, TypedDict

from rmm.pylibrmm.cuda_stream_pool import CudaStreamPool
from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from rapidsmpf.config import Options
from rapidsmpf.memory.buffer import Buffer, MemoryType
from rapidsmpf.memory.host_memory_resource import HostMemoryResource
from rapidsmpf.memory.memory_reservation import MemoryReservation
from rapidsmpf.memory.pinned_memory_resource import (
    PinnedMemoryResource,
    PinnedPoolProperties,
)
from rapidsmpf.memory.spill_manager import SpillManager
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.statistics import Statistics

_ReservationAttributeValue = str | int | bool | float

class ReservationRequestPolicy(IntEnum):
    IMMEDIATE = ...
    OVERBOOK = ...
    SPILL = ...
    WAIT = ...
    WAIT_OR_OVERBOOK = ...
    WAIT_OR_FAIL = ...

class ReservationResizeReason(IntEnum):
    CONSUMED = ...
    SPLIT_OFF = ...

class ReservationReleaseReason(IntEnum):
    CLEARED = ...
    RELEASED_CONSUMED = ...
    RELEASED_SPLIT = ...

class _ReservationEventBase(TypedDict):
    sequence: int
    steady_time_ns: int

class _RequestSubmittedEvent(_ReservationEventBase):
    kind: Literal["request_submitted"]
    id: int
    size: int
    memory_types: list[MemoryType]
    policy: ReservationRequestPolicy
    net_memory_delta: int | None
    attributes: dict[str, _ReservationAttributeValue]

class _RequestWaitingEvent(_ReservationEventBase):
    kind: Literal["request_waiting"]
    id: int

class _RequestGrantedEvent(_ReservationEventBase):
    kind: Literal["request_granted"]
    id: int
    reservation_id: int | None
    memory_type: MemoryType
    size: int
    overbooked_bytes: int
    timed_out: bool

class _RequestEmptyEvent(_ReservationEventBase):
    kind: Literal["request_empty"]
    id: int
    timed_out: bool

class _RequestFailedEvent(_ReservationEventBase):
    kind: Literal["request_failed"]
    id: int
    error: str

class _RequestCancelledEvent(_ReservationEventBase):
    kind: Literal["request_cancelled"]
    id: int

class _ReservationCreatedEvent(_ReservationEventBase):
    kind: Literal["reservation_created"]
    id: int
    memory_type: MemoryType
    size: int
    attributes: dict[str, _ReservationAttributeValue]
    request_id: int | None
    parent_id: int | None

class _ReservationResizedEvent(_ReservationEventBase):
    kind: Literal["reservation_resized"]
    id: int
    size: int
    reason: ReservationResizeReason

class _ReservationReleasedEvent(_ReservationEventBase):
    kind: Literal["reservation_released"]
    id: int
    reason: ReservationReleaseReason

class _MemoryLimitChangedEvent(_ReservationEventBase):
    kind: Literal["memory_limit_changed"]
    memory_type: MemoryType
    configured_limit: int

_ReservationEvent = (
    _RequestSubmittedEvent
    | _RequestWaitingEvent
    | _RequestGrantedEvent
    | _RequestEmptyEvent
    | _RequestFailedEvent
    | _RequestCancelledEvent
    | _ReservationCreatedEvent
    | _ReservationResizedEvent
    | _ReservationReleasedEvent
    | _MemoryLimitChangedEvent
)

class BufferResource:
    def __init__(
        self,
        device_mr: DeviceMemoryResource,
        *,
        pinned_pool_properties: PinnedPoolProperties | None = None,
        memory_limits: Mapping[MemoryType, int] | None = None,
        periodic_spill_check: float | None = 1e-3,
        stream_pool: CudaStreamPool | None = None,
        statistics: Statistics | None = None,
        spill_directory: str | os.PathLike[str] | None = None,
        memory_reserve_timeout: float = 0.1,
        allow_overbooking_by_default: bool = True,
        reservation_event_limit: int | None = None,
    ) -> None: ...
    @classmethod
    def from_options(
        cls: type[Self],
        mr: DeviceMemoryResource,
        options: Options,
        statistics: Statistics | None = None,
    ) -> Self: ...
    @property
    def device_mr(self) -> OwningDeviceMemoryResource: ...
    def device_mr_adaptor(self) -> RmmResourceAdaptor: ...
    @property
    def host_mr(self) -> HostMemoryResource: ...
    @property
    def spill_directory(self) -> pathlib.Path | None: ...
    @property
    def memory_reserve_timeout(self) -> float: ...
    @property
    def allow_overbooking_by_default(self) -> bool: ...
    @property
    def reservation_event_limit(self) -> int | None: ...
    @property
    def pinned_mr(self) -> PinnedMemoryResource | None: ...
    def memory_available(self, mem_type: MemoryType) -> int: ...
    def memory_available_for_reservation(self, mem_type: MemoryType) -> int: ...
    def memory_limit(self, mem_type: MemoryType) -> int: ...
    def has_memory_resource(self, mem_type: MemoryType) -> bool: ...
    def set_memory_limit(self, mem_type: MemoryType, limit: int) -> None: ...
    def start_reservation_event_recording(self) -> None: ...
    def drain_reservation_events(self) -> list[_ReservationEvent]: ...
    def stop_reservation_event_recording(self) -> list[_ReservationEvent]: ...
    @property
    def spill_manager(self) -> SpillManager: ...
    @property
    def statistics(self) -> Statistics: ...
    def reserve(
        self,
        mem_type: MemoryType,
        size: int,
        *,
        allow_overbooking: bool,
        attributes: Mapping[str, str | int | bool | float] | None = None,
    ) -> tuple[MemoryReservation, int]: ...
    def reserve_device_memory_and_spill(
        self,
        size: int,
        *,
        allow_overbooking: bool,
        attributes: Mapping[str, str | int | bool | float] | None = None,
    ) -> MemoryReservation: ...
    def reserve_or_fail(
        self,
        size: int,
        mem_types: list[MemoryType],
        *,
        attributes: Mapping[str, str | int | bool | float] | None = None,
    ) -> MemoryReservation: ...
    def release(self, reservation: MemoryReservation, size: int) -> int: ...
    def make_buffer(
        self, size: int, stream: Stream, reservation: MemoryReservation
    ) -> Buffer: ...
    @property
    def stream_pool(self) -> CudaStreamPool: ...

class OwningDeviceMemoryResource(DeviceMemoryResource): ...

def device_limit_from_options(options: Options) -> int: ...
def host_limit_from_options(options: Options) -> int | None: ...
def periodic_spill_check_from_options(options: Options) -> float | None: ...
def memory_reserve_timeout_from_options(options: Options) -> float: ...
def allow_overbooking_by_default_from_options(options: Options) -> bool: ...
def reservation_event_limit_from_options(options: Options) -> int | None: ...
def spill_dir_from_options(options: Options) -> pathlib.Path | None: ...
def stream_pool_from_options(options: Options) -> CudaStreamPool: ...
