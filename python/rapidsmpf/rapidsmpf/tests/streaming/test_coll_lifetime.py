# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of ``aclose()`` and ``async with`` on the streaming collectives."""

from __future__ import annotations

import asyncio
import contextlib
import gc
from typing import TYPE_CHECKING, Any

import pytest

from rapidsmpf.streaming.coll import Collective
from rapidsmpf.streaming.coll.allgather import AllGather
from rapidsmpf.streaming.coll.shuffler import ShufflerAsync
from rapidsmpf.streaming.coll.sparse_alltoall import SparseAlltoall
from rapidsmpf.testing import make_partition_data, validate_partition_data

if TYPE_CHECKING:
    from collections.abc import Callable

    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context


def barrier() -> None:
    from mpi4py import MPI

    MPI.COMM_WORLD.barrier()


def ring_peers(comm: Communicator) -> tuple[list[int], list[int]]:
    if comm.nranks == 1:
        return [], []
    return [(comm.rank - 1) % comm.nranks], [(comm.rank + 1) % comm.nranks]


def make_shuffler(ctx: Context, comm: Communicator, op_id: int) -> ShufflerAsync:
    return ShufflerAsync(ctx, comm, op_id, comm.nranks)


def make_sparse_alltoall(
    ctx: Context, comm: Communicator, op_id: int
) -> SparseAlltoall:
    srcs, dsts = ring_peers(comm)
    return SparseAlltoall(ctx, comm, op_id, srcs, dsts)


def make_allgather(ctx: Context, comm: Communicator, op_id: int) -> AllGather:
    return AllGather(ctx, comm, op_id)


FACTORIES = {
    "shuffler": make_shuffler,
    "sparse_alltoall": make_sparse_alltoall,
    "allgather": make_allgather,
}
ASYNC_WITH_FACTORIES = {
    "shuffler": make_shuffler,
    "sparse_alltoall": make_sparse_alltoall,
}


async def insert_finished(coll: Any) -> None:
    if isinstance(coll, AllGather):
        coll.insert_finished()
    else:
        await coll.insert_finished()


async def wait_complete(coll: Any) -> None:
    await insert_finished(coll)
    if isinstance(coll, AllGather):
        await coll.extract_all(ordered=True)


async def assert_closed(coll: Any) -> None:
    name = type(coll).__name__
    with pytest.raises(RuntimeError, match=f"{name} is closed"):
        await insert_finished(coll)
    if isinstance(coll, ShufflerAsync):
        with pytest.raises(RuntimeError, match="is closed"):
            coll.local_partitions()
        with pytest.raises(RuntimeError, match="is closed"):
            coll.extract(0)
        with pytest.raises(RuntimeError, match="is closed"):
            coll.insert({})
    elif isinstance(coll, SparseAlltoall):
        with pytest.raises(RuntimeError, match="is closed"):
            coll.extract(0)
    else:
        with pytest.raises(RuntimeError, match="is closed"):
            await coll.extract_all(ordered=True)


def test_collective_base_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="abstract base class"):
        Collective()


@pytest.mark.parametrize("factory", FACTORIES.values(), ids=FACTORIES.keys())
def test_aclose(
    context: Context, comm: Communicator, factory: Callable[..., Any]
) -> None:
    async def run() -> None:
        coll = factory(context, comm, 0)
        assert isinstance(coll, Collective)
        await insert_finished(coll)
        await coll.aclose()
        await coll.aclose()
        await assert_closed(coll)

    asyncio.run(run())


@pytest.mark.parametrize("factory", FACTORIES.values(), ids=FACTORIES.keys())
def test_aclose_in_new_event_loop(
    context: Context, comm: Communicator, factory: Callable[..., Any]
) -> None:
    coll = factory(context, comm, 0)
    asyncio.run(insert_finished(coll))
    asyncio.run(coll.aclose())
    asyncio.run(assert_closed(coll))


@pytest.mark.parametrize("factory", FACTORIES.values(), ids=FACTORIES.keys())
def test_aclose_before_insert_finished(
    context: Context, comm: Communicator, factory: Callable[..., Any]
) -> None:
    async def run() -> None:
        coll = factory(context, comm, 0)
        with pytest.raises(RuntimeError, match="before insert_finished"):
            await coll.aclose()
        await coll.aclose()
        await assert_closed(coll)

    asyncio.run(run())


@pytest.mark.parametrize("factory", FACTORIES.values(), ids=FACTORIES.keys())
def test_aclose_before_insert_finished_on_one_rank(
    context: Context, comm: Communicator, factory: Callable[..., Any]
) -> None:
    if comm.nranks < 2:
        pytest.skip("Need at least two ranks")

    async def run() -> None:
        coll = factory(context, comm, 0)
        if comm.rank == 0:
            with pytest.raises(RuntimeError, match="before insert_finished"):
                await coll.aclose()
        else:
            await insert_finished(coll)
            await coll.aclose()
        await assert_closed(coll)

    asyncio.run(run())


@pytest.mark.parametrize(
    "factory", ASYNC_WITH_FACTORIES.values(), ids=ASYNC_WITH_FACTORIES.keys()
)
def test_async_with(
    context: Context, comm: Communicator, factory: Callable[..., Any]
) -> None:
    async def run() -> None:
        async with factory(context, comm, 0) as coll:
            await coll.insert_finished()
        await assert_closed(coll)

    asyncio.run(run())


@pytest.mark.parametrize(
    "factory", ASYNC_WITH_FACTORIES.values(), ids=ASYNC_WITH_FACTORIES.keys()
)
def test_async_with_body_raises(
    context: Context, comm: Communicator, factory: Callable[..., Any]
) -> None:
    class BodyError(Exception):
        pass

    async def run() -> None:
        coll = factory(context, comm, 0)
        with pytest.raises(BodyError):
            async with coll:
                raise BodyError
        await assert_closed(coll)

    asyncio.run(run())


@pytest.mark.parametrize(
    "factory", ASYNC_WITH_FACTORIES.values(), ids=ASYNC_WITH_FACTORIES.keys()
)
def test_async_with_without_insert_finished(
    context: Context, comm: Communicator, factory: Callable[..., Any]
) -> None:
    async def run() -> None:
        coll = factory(context, comm, 0)
        with pytest.raises(RuntimeError, match="before insert_finished"):
            async with coll:
                pass
        await assert_closed(coll)

    asyncio.run(run())


def test_shuffler_op_id_reuse_after_aclose(
    context: Context, comm: Communicator, stream: Stream
) -> None:
    total_num_partitions = 5
    total_num_rows = 100

    async def shuffle(base: int) -> None:
        async with ShufflerAsync(context, comm, 0, total_num_partitions) as shuffler:
            for pid in shuffler.local_partitions():
                chunks = make_partition_data(
                    total_num_partitions,
                    total_num_rows,
                    pid,
                    stream,
                    context.br(),
                    base=base,
                )
                if chunks:
                    shuffler.insert(chunks)
            await shuffler.insert_finished()
            for pid in shuffler.local_partitions():
                validate_partition_data(
                    shuffler.extract(pid),
                    total_num_partitions,
                    total_num_rows,
                    pid,
                    base=base,
                )

    async def run() -> None:
        await shuffle(42)
        await shuffle(123)

    asyncio.run(run())


@pytest.mark.parametrize("factory", FACTORIES.values(), ids=FACTORIES.keys())
def test_cancelled_aclose_with_stuck_peer(
    context: Context, comm: Communicator, factory: Callable[..., Any]
) -> None:
    if comm.nranks < 2:
        pytest.skip("Need at least two ranks")

    async def run() -> None:
        coll = factory(context, comm, 0)
        if comm.rank == 0:
            # Rank 1 inserts its finish markers only after the barrier, so the
            # collective cannot complete before the timeout.
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(coll.aclose(), timeout=0.1)
            barrier()
            with pytest.raises(RuntimeError, match="before insert_finished"):
                await coll.aclose()
        else:
            barrier()
            await insert_finished(coll)
            await coll.aclose()
        await assert_closed(coll)

    asyncio.run(run())


@pytest.mark.parametrize(
    "factory,op_id",
    [(f, i) for i, f in enumerate(FACTORIES.values(), start=10)],
    ids=FACTORIES.keys(),
)
def test_drop_while_waiting(
    context: Context, comm: Communicator, factory: Callable[..., Any], op_id: int
) -> None:
    """
    Drop a collective while a background task still waits for it to complete.

    On rank 0 the task waits until rank 1 inserts its finish markers after the
    barrier. On a single rank the collective may already have completed when the
    wait is cancelled, so the in-flight path is not always exercised.
    """

    async def run() -> None:
        # Each collective gets its own op_id, since rank 0 cannot tell when the
        # dropped collective has completed. If it has not completed when the context
        # shuts down, it leaks for the rest of the session and keeps its op_id on
        # the session communicator, so no other test may use op_ids 10 to 12.
        coll = factory(context, comm, op_id)
        if comm.rank == 0:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(wait_complete(coll), timeout=0.1)
            del coll
            gc.collect()
            barrier()
        else:
            barrier()
            await insert_finished(coll)
            await coll.aclose()

    asyncio.run(run())
