# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.memory.reservation_events import ReservationContext
from rapidsmpf.streaming.chunks.packed_data import PackedDataChunk
from rapidsmpf.streaming.coll.collective import Collective
from rapidsmpf.streaming.core.actor import CppActor
from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context

class AllGather(Collective):
    def __init__(
        self,
        ctx: Context,
        comm: Communicator,
        op_id: int,
        *,
        reservation_context: ReservationContext | None = None,
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
    async def extract_all(
        self, ctx: Context | None = None, *, ordered: bool
    ) -> list[PackedData]: ...

def allgather(
    ctx: Context,
    comm: Communicator,
    ch_in: Channel[PackedDataChunk],
    ch_out: Channel[PackedDataChunk],
    op_id: int,
    *,
    ordered: bool,
    reservation_context: ReservationContext | None = None,
) -> CppActor: ...
