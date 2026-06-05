# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.coll.shuffler import ShufflerAsync

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context


async def _roundtrip(context: Context, comm: Communicator) -> None:
    packed = PackedData.from_host_bytes(b"hello world", context.br())
    shuffler = ShufflerAsync(
        context,
        comm,
        op_id=2,
        total_num_partitions=1,
    )

    shuffler.insert({0: packed})
    await shuffler.insert_finished(context)

    result = shuffler.extract(0)
    assert len(result) == 1
    assert result[0].to_host_bytes() == b"hello world"


def test_streaming_shuffler_packed_data_roundtrip(
    context: Context,
    comm: Communicator,
) -> None:
    asyncio.run(_roundtrip(context, comm))
