# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.statistics import Statistics

class CollectiveKind(IntEnum):
    ALLGATHER = ...
    SPARSE_ALLTOALL = ...
    SHUFFLER = ...
    ALLREDUCE = ...

class TransferEvent(NamedTuple):
    op_id: int
    collective_kind: CollectiveKind
    source_rank: int
    destination_rank: int
    message_id: int
    metadata_bytes: int
    payload_bytes: int
    destination_memory_type: MemoryType
    completion_timestamp_ns: int

class ProgressThread:
    def __init__(self, statistics: Statistics | None = None) -> None: ...
    @property
    def statistics(self) -> Statistics: ...
    def enable_transfer_events(self, capacity: int = 65536) -> None: ...
    def disable_transfer_events(self) -> None: ...
    def drain_transfer_events(self) -> list[TransferEvent]: ...
    @property
    def dropped_transfer_events(self) -> int: ...
