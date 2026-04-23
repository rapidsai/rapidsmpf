# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable

from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.packed_data import PackedData

class SparseAlltoall:
    def __init__(
        self,
        comm: Communicator,
        op_id: int,
        br: BufferResource,
        srcs: Iterable[int],
        dsts: Iterable[int],
    ) -> None: ...
    @property
    def comm(self) -> Communicator: ...
    def insert(self, dst: int, packed_data: PackedData) -> None: ...
    def insert_finished(self) -> None: ...
    def wait(self, timeout_ms: int = -1) -> None: ...
    def extract(self, src: int) -> list[PackedData]: ...
