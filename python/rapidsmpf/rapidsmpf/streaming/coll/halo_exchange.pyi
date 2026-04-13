# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.core.context import Context

class HaloExchange:
    def __init__(self, ctx: Context, comm: Communicator, op_id: int) -> None: ...
    async def exchange(
        self,
        send_left: PackedData | None,
        send_right: PackedData | None,
    ) -> tuple[PackedData | None, PackedData | None]: ...
