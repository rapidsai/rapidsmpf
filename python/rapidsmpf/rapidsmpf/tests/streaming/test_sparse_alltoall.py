# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import pytest

import pylibcudf as plc

from rapidsmpf.integrations.cudf.partition import unpack_and_concat
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.coll.sparse_alltoall import SparseAlltoall
from rapidsmpf.testing import assert_eq

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context


def make_packed_data(context: Context, values: np.ndarray) -> PackedData:
    stream = context.get_stream_from_pool()
    table = plc.Table([plc.Column.from_array(values, stream=stream)])
    return PackedData.from_cudf_packed_columns(
        plc.contiguous_split.pack(table, stream=stream),
        stream,
        context.br(),
    )


def unpack_table(context: Context, packed_data: PackedData) -> plc.Table:
    stream = context.get_stream_from_pool()
    return unpack_and_concat([packed_data], stream, context.br())


def test_sparse_alltoall_non_participating_ranks(
    context: Context,
    comm: Communicator,
) -> None:
    if comm.nranks == 0:
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
        exchange.insert(1, make_packed_data(context, np.array([11], dtype=np.int32)))
        exchange.insert(1, make_packed_data(context, np.array([29], dtype=np.int32)))

    asyncio.run(exchange.insert_finished(context))

    if comm.rank == 1:
        results = exchange.extract(0)
        assert len(results) == 2
        stream = context.get_stream_from_pool()
        assert_eq(
            unpack_table(context, results[0]),
            plc.Table(
                [plc.Column.from_array(np.array([11], dtype=np.int32), stream=stream)]
            ),
        )
        assert_eq(
            unpack_table(context, results[1]),
            plc.Table(
                [plc.Column.from_array(np.array([29], dtype=np.int32), stream=stream)]
            ),
        )


def test_sparse_alltoall_invalid_constructor(
    context: Context, comm: Communicator
) -> None:
    rank = comm.rank
    size = comm.nranks
    for src, dst in [([], [rank]), ([rank], []), ([], [size]), ([size], [])]:
        with pytest.raises(
            RuntimeError, match=r"SparseAlltoall invalid (source|destination) rank"
        ):
            SparseAlltoall(context, comm, 1, src, dst)
    if size > 1:
        for src, dst in [([], [(rank + 1) % size]), ([(rank + 1) % size], [])]:
            with pytest.raises(
                RuntimeError,
                match=r"SparseAlltoall (source|destination) rank list must be unique",
            ):
                SparseAlltoall(context, comm, 1, src, dst)
