# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import pytest

import pylibcudf as plc
import rmm.mr

from rapidsmpf.communicator import COMMUNICATORS
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.integrations.cudf.partition import unpack_and_concat
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.streaming.coll.sparse_alltoall import SparseAlltoall
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.testing import assert_eq

if TYPE_CHECKING:
    from collections.abc import Generator

    from rapidsmpf.communicator.communicator import Communicator


@pytest.fixture(params=["mpi", "ucxx"], scope="module")
def distributed_comm(
    request: pytest.FixtureRequest,
) -> Generator[Communicator, None, None]:
    comm_name = request.param

    if "mpi" not in COMMUNICATORS:
        if comm_name == "mpi":
            pytest.skip("RapidsMPF not built with MPI support")
        pytest.skip(
            "RapidsMPF not built with MPI support, which is used to bootstrap UCXX"
        )
    if comm_name == "ucxx" and "ucxx" not in COMMUNICATORS:
        pytest.skip("RapidsMPF not built with UCXX support")

    from mpi4py import MPI

    MPI.COMM_WORLD.barrier()
    yield request.getfixturevalue(f"_{comm_name}_comm")
    MPI.COMM_WORLD.barrier()


@pytest.fixture(scope="module")
def distributed_context(
    distributed_comm: Communicator,
) -> Generator[Context, None, None]:
    options = Options(get_environment_variables())
    mr = RmmResourceAdaptor(rmm.mr.CudaMemoryResource())
    br = BufferResource(mr)

    with Context(distributed_comm.logger, br, options) as ctx:
        yield ctx


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


def require_multiple_ranks(comm: Communicator) -> None:
    if comm.nranks == 1:
        pytest.skip("Need at least two ranks")


def test_sparse_alltoall_non_participating_ranks(
    distributed_context: Context,
    distributed_comm: Communicator,
) -> None:
    require_multiple_ranks(distributed_comm)

    if distributed_comm.rank == 0:
        srcs = []
        dsts = [1]
    elif distributed_comm.rank == 1:
        srcs = [0]
        dsts = []
    else:
        srcs = []
        dsts = []

    exchange = SparseAlltoall(
        distributed_context,
        distributed_comm,
        0,
        srcs,
        dsts,
    )

    if distributed_comm.rank == 0:
        exchange.insert(
            1, make_packed_data(distributed_context, np.array([11], dtype=np.int32))
        )
        exchange.insert(
            1, make_packed_data(distributed_context, np.array([29], dtype=np.int32))
        )

    asyncio.run(exchange.insert_finished(distributed_context))

    if distributed_comm.rank == 1:
        results = exchange.extract(0)
        assert len(results) == 2
        stream = distributed_context.get_stream_from_pool()
        assert_eq(
            unpack_table(distributed_context, results[0]),
            plc.Table(
                [plc.Column.from_array(np.array([11], dtype=np.int32), stream=stream)]
            ),
        )
        assert_eq(
            unpack_table(distributed_context, results[1]),
            plc.Table(
                [plc.Column.from_array(np.array([29], dtype=np.int32), stream=stream)]
            ),
        )


@pytest.mark.parametrize("invalid_srcs", [True, False])
def test_sparse_alltoall_invalid_constructor(
    distributed_context: Context,
    distributed_comm: Communicator,
    invalid_srcs: bool,  # noqa: FBT001
) -> None:
    require_multiple_ranks(distributed_comm)

    valid_other = (distributed_comm.rank + 1) % distributed_comm.nranks
    valid_peers = [] if valid_other == distributed_comm.rank else [valid_other]
    invalid_peers = [distributed_comm.rank]
    srcs = invalid_peers if invalid_srcs else valid_peers
    dsts = valid_peers if invalid_srcs else invalid_peers

    with pytest.raises(
        RuntimeError, match=r"SparseAlltoall invalid (source|destination) rank"
    ):
        SparseAlltoall(distributed_context, distributed_comm, 1, srcs, dsts)
