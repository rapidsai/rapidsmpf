# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Single-rank (smoke) tests for HaloExchange.

Because the streaming conftest always uses single_process_comm (nranks=1),
these tests exercise the code path but with trivial results: boundary
conditions eliminate both neighbors, so exchange() always returns (None, None).

Multi-rank tests live in ../test_halo_exchange.py, which uses the MPI comm
fixture and is run via ``mpirun -np N``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import pylibcudf as plc

from rapidsmpf.integrations.cudf.partition import unpack_and_concat
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.coll.halo_exchange import HaloExchange
from rapidsmpf.streaming.core.actor import define_actor, run_actor_network

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context


def _make_packed_data(ctx: Context, values: list[int]) -> PackedData:
    stream = ctx.get_stream_from_pool()
    table = plc.Table([plc.Column.from_array(np.array(values, dtype=np.int32), stream=stream)])
    return PackedData.from_cudf_packed_columns(
        plc.contiguous_split.pack(table, stream=stream),
        stream,
        ctx.br(),
    )


@define_actor()
async def _exchange_actor(
    context: Context,
    comm: Communicator,
    op_id: int,
    send_right: PackedData | None,
    send_left: PackedData | None,
) -> None:
    he = HaloExchange(context, comm, op_id)
    from_left, from_right = await he.exchange(send_right, send_left)
    # nranks==1: no neighbors, so both directions are always None
    assert from_left is None
    assert from_right is None


def test_single_rank_no_data(
    context: Context, comm: Communicator, py_executor: ThreadPoolExecutor
) -> None:
    """exchange(None, None) with nranks=1 returns (None, None)."""
    actor = _exchange_actor(context, comm, op_id=0, send_right=None, send_left=None)
    run_actor_network(actors=[actor], py_executor=py_executor)


def test_single_rank_with_data(
    context: Context, comm: Communicator, py_executor: ThreadPoolExecutor
) -> None:
    """exchange(data, None) with nranks=1 still returns (None, None) — no neighbors."""
    data = _make_packed_data(context, [1, 2, 3])
    actor = _exchange_actor(context, comm, op_id=0, send_right=data, send_left=None)
    run_actor_network(actors=[actor], py_executor=py_executor)


@define_actor()
async def _multi_round_actor(
    context: Context,
    comm: Communicator,
    op_id: int,
    n_rounds: int,
) -> None:
    he = HaloExchange(context, comm, op_id)
    for _ in range(n_rounds):
        from_left, from_right = await he.exchange(None, None)
        assert from_left is None
        assert from_right is None


def test_multi_round_same_instance(
    context: Context, comm: Communicator, py_executor: ThreadPoolExecutor
) -> None:
    """Repeated exchange() calls on the same instance complete without error."""
    actor = _multi_round_actor(context, comm, op_id=0, n_rounds=5)
    run_actor_network(actors=[actor], py_executor=py_executor)
