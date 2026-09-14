# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

from rapidsmpf.memory.buffer import MemoryType

class SpillReason(IntEnum):
    PERIODIC = 0
    RESERVATION = 1
    EAGER = 2
    EXPLICIT = 3

@dataclass(frozen=True)
class SpillAttempt:
    event_id: int
    attempt_id: int
    start_ns: int
    requested_bytes: int
    reason: SpillReason
    is_background: bool
    evictor: int | None

@dataclass(frozen=True)
class SpillTransfer:
    event_id: int
    attempt_id: int
    submission_start_ns: int
    submission_end_ns: int
    submitted_bytes: int
    source: MemoryType
    destination: MemoryType
    reason: SpillReason
    is_background: bool
    evictor: int | None
    buffer_owner: int | None
    is_success: bool

class SpillManager:
    def add_spill_function(
        self,
        func: Callable[[int], int],
        priority: int,
        buffer_owner: int | None = None,
    ) -> int: ...
    def remove_spill_function(self, function_id: int) -> None: ...
    def spill(
        self,
        amount: int,
        reason: SpillReason = SpillReason.EXPLICIT,
        evictor: int | None = None,
    ) -> int: ...
    def enable_event_collection(self, capacity: int = 4096) -> None: ...
    def disable_event_collection(self) -> None: ...
    @property
    def event_collection_enabled(self) -> bool: ...
    @property
    def event_sequence(self) -> int: ...
    @property
    def dropped_events(self) -> int: ...
    def read_spill_attempts(
        self, begin_sequence: int, end_sequence: int | None = None
    ) -> list[SpillAttempt]: ...
    def read_spill_transfers(
        self, begin_sequence: int, end_sequence: int | None = None
    ) -> list[SpillTransfer]: ...
    def spill_to_make_headroom(self, headroom: int = 0) -> int: ...
