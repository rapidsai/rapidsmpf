# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-rank tests for HaloExchange.

Run via ``mpirun -np N python -m pytest tests/test_halo_exchange.py``.
This module uses the root conftest's MPI/UCXX ``comm`` fixture so that
``comm.nranks == N`` and each process has a distinct rank.

Tests that require exactly 2 ranks skip for all other N values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import pylibcudf as plc
import rmm.mr

from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.integrations.cudf.partition import unpack_and_concat
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.streaming.coll.halo_exchange import HaloExchange
from rapidsmpf.streaming.core.actor import define_actor, run_actor_network
from rapidsmpf.streaming.core.context import Context

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    from rapidsmpf.communicator.communicator import Communicator


@pytest.fixture
def streaming_context(comm: Communicator):
    """Streaming context backed by the test's MPI/UCXX communicator."""
    options = Options(get_environment_variables())
    mr = RmmResourceAdaptor(rmm.mr.CudaMemoryResource())
    br = BufferResource(mr)
    with Context(comm.logger, br, options) as ctx:
        yield ctx


@pytest.fixture(scope="session")
def py_executor():
    from concurrent.futures import ThreadPoolExecutor
    return ThreadPoolExecutor(max_workers=1)


def _make_packed_data(ctx: Context, values: list[int]) -> PackedData:
    stream = ctx.get_stream_from_pool()
    table = plc.Table([plc.Column.from_array(np.array(values, dtype=np.int32), stream=stream)])
    return PackedData.from_cudf_packed_columns(
        plc.contiguous_split.pack(table, stream=stream),
        stream,
        ctx.br(),
    )


def _unpack_to_list(ctx: Context, pd: PackedData) -> list[int]:
    stream = ctx.get_stream_from_pool()
    table = unpack_and_concat([pd], stream, ctx.br())
    stream.synchronize()
    return table.columns()[0].to_array().tolist()


# ---------------------------------------------------------------------------
# Two-rank tests
# ---------------------------------------------------------------------------


@define_actor()
async def _two_rank_exchange_actor(
    context: Context,
    comm: Communicator,
    op_id: int,
) -> None:
    """
    Rank 0 sends [10, 20, 30] rightward; rank 1 sends [40, 50] leftward.
    Expected: rank 0 gets from_right=[40, 50]; rank 1 gets from_left=[10, 20, 30].
    """
    rank = comm.rank
    he = HaloExchange(context, comm, op_id)

    if rank == 0:
        data = _make_packed_data(context, [10, 20, 30])
        from_left, from_right = await he.exchange(send_right=data, send_left=None)
        assert from_left is None
        assert from_right is not None
        assert _unpack_to_list(context, from_right) == [40, 50]
    else:  # rank == 1
        data = _make_packed_data(context, [40, 50])
        from_left, from_right = await he.exchange(send_right=None, send_left=data)
        assert from_right is None
        assert from_left is not None
        assert _unpack_to_list(context, from_left) == [10, 20, 30]


def test_two_rank_exchange(
    streaming_context: Context,
    comm: Communicator,
    py_executor: ThreadPoolExecutor,
) -> None:
    """Rank 0 sends right, rank 1 sends left — each receives the other's data."""
    if comm.nranks != 2:
        pytest.skip("Requires exactly 2 ranks")

    actor = _two_rank_exchange_actor(streaming_context, comm, op_id=0)
    run_actor_network(actors=[actor], py_executor=py_executor)


@define_actor()
async def _two_rank_multi_round_actor(
    context: Context,
    comm: Communicator,
    op_id: int,
    n_rounds: int,
) -> None:
    """Two ranks exchange scalar values across n_rounds on the same HaloExchange instance."""
    rank = comm.rank
    he = HaloExchange(context, comm, op_id)

    for i in range(n_rounds):
        value = rank * 100 + i
        data = _make_packed_data(context, [value])

        if rank == 0:
            from_left, from_right = await he.exchange(send_right=data, send_left=None)
            assert from_left is None
            assert from_right is not None
            expected = [100 + i]  # rank 1's value for this round
            assert _unpack_to_list(context, from_right) == expected, (
                f"Round {i}: rank 0 from_right expected {expected}"
            )
        else:  # rank == 1
            from_left, from_right = await he.exchange(send_right=None, send_left=data)
            assert from_right is None
            assert from_left is not None
            expected = [i]  # rank 0's value for this round
            assert _unpack_to_list(context, from_left) == expected, (
                f"Round {i}: rank 1 from_left expected {expected}"
            )


def test_two_rank_multi_round(
    streaming_context: Context,
    comm: Communicator,
    py_executor: ThreadPoolExecutor,
) -> None:
    """Multiple exchange() rounds on the same instance produce correct results."""
    if comm.nranks != 2:
        pytest.skip("Requires exactly 2 ranks")

    actor = _two_rank_multi_round_actor(streaming_context, comm, op_id=0, n_rounds=3)
    run_actor_network(actors=[actor], py_executor=py_executor)


# ---------------------------------------------------------------------------
# Boundary rank tests (3+ ranks)
# ---------------------------------------------------------------------------


@define_actor()
async def _boundary_ranks_actor(
    context: Context,
    comm: Communicator,
    op_id: int,
) -> None:
    """
    With nranks >= 2: rank 0 has no left neighbor, rank nranks-1 has no right neighbor.
    Interior ranks exchange in both directions.
    """
    rank = comm.rank
    nranks = comm.nranks
    he = HaloExchange(context, comm, op_id)

    send_right = _make_packed_data(context, [rank]) if rank < nranks - 1 else None
    send_left = _make_packed_data(context, [rank]) if rank > 0 else None

    from_left, from_right = await he.exchange(send_right=send_right, send_left=send_left)

    # Boundary invariants
    if rank == 0:
        assert from_left is None, "rank 0 must have from_left=None"
    if rank == nranks - 1:
        assert from_right is None, f"rank {rank} (last) must have from_right=None"

    # Interior / non-boundary receives
    if rank > 0:
        assert from_left is not None
        assert _unpack_to_list(context, from_left) == [rank - 1]
    if rank < nranks - 1:
        assert from_right is not None
        assert _unpack_to_list(context, from_right) == [rank + 1]


def test_boundary_ranks(
    streaming_context: Context,
    comm: Communicator,
    py_executor: ThreadPoolExecutor,
) -> None:
    """
    All ranks exchange with neighbors; boundary ranks correctly receive None
    for the absent-neighbor direction.
    """
    if comm.nranks < 2:
        pytest.skip("Requires at least 2 ranks")

    actor = _boundary_ranks_actor(streaming_context, comm, op_id=0)
    run_actor_network(actors=[actor], py_executor=py_executor)
