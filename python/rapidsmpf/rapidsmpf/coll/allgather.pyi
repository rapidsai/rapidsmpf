# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.packed_data import PackedData

class AllGather:
    def __init__(
        self,
        comm: Communicator,
        op_id: int,
        br: BufferResource,
    ) -> None: ...
    @property
    def comm(self) -> Communicator: ...
    def __enter__(self) -> AllGather: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ) -> bool: ...
    def insert(self, sequence_number: int, packed_data: PackedData) -> None: ...
    def insert_finished(self) -> None: ...
    def wait_and_extract(
        self, ordered: bool = True, timeout_ms: int = -1
    ) -> list[PackedData]: ...
