# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from rapidsmpf.streaming.coll.sparse_alltoall import SparseAlltoall
from rapidsmpf.testing import generate_packed_data, validate_packed_data

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.packed_data import PackedData
    from rapidsmpf.streaming.core.context import Context


def make_packed_data(context: Context, value: int) -> PackedData:
    stream = context.br().stream_pool.get_stream()
    return generate_packed_data(1, value, stream, context.br())


def test_sparse_alltoall_non_participating_ranks(
    context: Context,
    comm: Communicator,
) -> None:
    if comm.nranks < 2:
        pytest.skip("Need at least two ranks")
    if comm.rank == 0:
        srcs = []
        dsts = [1]
    elif comm.rank == 1:
        srcs = [0]
        dsts = []
    else:
        srcs = []
        dsts = []

    exchange = SparseAlltoall(
        context,
        comm,
        0,
        srcs,
        dsts,
    )

    if comm.rank == 0:
        exchange.insert(1, make_packed_data(context, 11))
        exchange.insert(1, make_packed_data(context, 29))

    asyncio.run(exchange.insert_finished(context))

    if comm.rank == 1:
        results = exchange.extract(0)
        assert len(results) == 2
        validate_packed_data(results[0], 1, 11)
        validate_packed_data(results[1], 1, 29)
