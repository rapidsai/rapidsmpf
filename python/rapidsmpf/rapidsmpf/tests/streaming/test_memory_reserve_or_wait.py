# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import pytest

import rmm.mr

from rapidsmpf.communicator.single import (
    new_communicator as single_process_comm,
)
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.buffer_resource import BufferResource, LimitAvailableMemory
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.streaming.core.actor import define_actor, run_actor_network
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.memory_reserve_or_wait import (
    MemoryReserveOrWait,
    reserve_memory,
)

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor


def make_context(
    *, dev_limit: int, overwrite_options: dict[str, str] | None = None
) -> Context:
    env = get_environment_variables()
    if overwrite_options is not None:
        env.update(overwrite_options)
    options = Options(env)
    comm = single_process_comm(options, ProgressThread())
    mr = RmmResourceAdaptor(rmm.mr.CudaMemoryResource())
    br = BufferResource(
        mr,
        memory_available={MemoryType.DEVICE: LimitAvailableMemory(mr, limit=dev_limit)},
    )
    return Context(comm, br, options)


def test_memory_is_available(py_executor: ThreadPoolExecutor) -> None:
    with make_context(dev_limit=1024) as context:
        mrow = MemoryReserveOrWait(context.options(), MemoryType.DEVICE, context)

        @define_actor()
        async def actor(ctx: Context) -> None:
            res = await mrow.reserve_or_wait(size=1024, net_memory_delta=0)
            assert res.mem_type == MemoryType.DEVICE
            assert res.size == 1024

        run_actor_network(
            actors=[actor(context)],
            py_executor=py_executor,
        )


def test_reserve_zero_is_always_available(py_executor: ThreadPoolExecutor) -> None:
    with make_context(dev_limit=0) as context:
        mrow = MemoryReserveOrWait(
            Options({"memory_reserve_timeout": "10m"}), MemoryType.DEVICE, context
        )

        @define_actor()
        async def actor(ctx: Context) -> None:
            t0 = time.time()
            res = await mrow.reserve_or_wait(size=0, net_memory_delta=0)
            assert time.time() - t0 < 10  #  Should complete before timeout.
            assert res.mem_type == MemoryType.DEVICE
            assert res.size == 0

        run_actor_network(
            actors=[actor(context)],
            py_executor=py_executor,
        )


def test_timeout(py_executor: ThreadPoolExecutor) -> None:
    with make_context(dev_limit=1024) as context:
        mrow = MemoryReserveOrWait(
            Options({"memory_reserve_timeout": "1ms"}), MemoryType.DEVICE, context
        )

        @define_actor()
        async def actor(ctx: Context) -> None:
            res = await mrow.reserve_or_wait(size=2048, net_memory_delta=0)
            assert res.mem_type == MemoryType.DEVICE
            assert res.size == 0

        run_actor_network(
            actors=[actor(context)],
            py_executor=py_executor,
        )


def test_shutdown(py_executor: ThreadPoolExecutor) -> None:
    with make_context(dev_limit=1024) as context:
        mrow = MemoryReserveOrWait(
            Options({"memory_reserve_timeouts": "10m"}), MemoryType.DEVICE, context
        )

        @define_actor()
        async def actor1(ctx: Context) -> None:
            with pytest.raises(RuntimeError, match="memory reservation failed"):
                await mrow.reserve_or_wait(size=2048, net_memory_delta=0)

        @define_actor()
        async def actor2(ctx: Context) -> None:
            # Wait until `actor1()` has submitted its reservation request.
            while mrow.size() == 0:
                await asyncio.sleep(0)
            await mrow.shutdown()

        run_actor_network(
            actors=[actor1(context), actor2(context)],
            py_executor=py_executor,
        )


def test_context_memory_returns_handle(py_executor: ThreadPoolExecutor) -> None:
    with make_context(dev_limit=1024) as context:
        mrow = context.memory(MemoryType.DEVICE)
        assert isinstance(mrow, MemoryReserveOrWait)

        @define_actor()
        async def actor(ctx: Context) -> None:
            assert ctx.memory(MemoryType.DEVICE) is mrow
            res = await mrow.reserve_or_wait(size=512, net_memory_delta=0)
            assert res.mem_type == MemoryType.DEVICE
            assert res.size == 512

        run_actor_network(
            actors=[actor(context)],
            py_executor=py_executor,
        )


def test_reserve_or_wait_or_overbook(py_executor: ThreadPoolExecutor) -> None:
    with make_context(dev_limit=2048) as context:
        mrow = MemoryReserveOrWait(
            Options({"memory_reserve_timeout": "1ms"}), MemoryType.DEVICE, context
        )

        @define_actor()
        async def actor(ctx: Context) -> None:
            # No overbooking
            res, overbooked = await mrow.reserve_or_wait_or_overbook(
                size=1024, net_memory_delta=0
            )
            assert res.mem_type == MemoryType.DEVICE
            assert res.size == 1024
            assert overbooked == 0

            # Some overbooking
            res, overbooked = await mrow.reserve_or_wait_or_overbook(
                size=2048, net_memory_delta=0
            )
            assert res.mem_type == MemoryType.DEVICE
            assert res.size == 2048
            assert overbooked == 1024

        run_actor_network(
            actors=[actor(context)],
            py_executor=py_executor,
        )


def test_reserve_or_wait_or_fail(py_executor: ThreadPoolExecutor) -> None:
    with make_context(dev_limit=1024) as context:
        mrow = MemoryReserveOrWait(
            Options({"memory_reserve_timeout": "1ms"}), MemoryType.DEVICE, context
        )

        @define_actor()
        async def actor(ctx: Context) -> None:
            # Request cannot be satisfied and overbooking is not allowed.
            with pytest.raises(RuntimeError):
                await mrow.reserve_or_wait_or_fail(size=2048, net_memory_delta=0)

        run_actor_network(
            actors=[actor(context)],
            py_executor=py_executor,
        )


def test_reserve_memory_helper(py_executor: ThreadPoolExecutor) -> None:
    with make_context(
        dev_limit=1024, overwrite_options={"memory_reserve_timeout": "1ms"}
    ) as context:

        @define_actor()
        async def actor(ctx: Context) -> None:
            # Fits within limit, should always succeed.
            res = await reserve_memory(
                ctx,
                512,
                net_memory_delta=0,
                mem_type=MemoryType.DEVICE,
                allow_overbooking=False,
            )
            assert res.mem_type == MemoryType.DEVICE
            assert res.size == 512

            # Exceeds limit, overbooking enabled, should succeed.
            res = await reserve_memory(
                ctx,
                2048,
                net_memory_delta=0,
                mem_type=MemoryType.DEVICE,
                allow_overbooking=True,
            )
            assert res.mem_type == MemoryType.DEVICE
            assert res.size == 2048

            # Exceeds limit, overbooking disabled, should fail.
            with pytest.raises(RuntimeError):
                await reserve_memory(
                    ctx,
                    2048,
                    net_memory_delta=0,
                    mem_type=MemoryType.DEVICE,
                    allow_overbooking=False,
                )

        run_actor_network(
            actors=[actor(context)],
            py_executor=py_executor,
        )


def test_reserve_memory_helper_allow_overbooking_by_default(
    py_executor: ThreadPoolExecutor,
) -> None:
    # Option enabled, should overbook and succeed.
    with make_context(
        dev_limit=1024,
        overwrite_options={
            "memory_reserve_timeout": "1ms",
            "allow_overbooking_by_default": "true",
        },
    ) as context:

        @define_actor()
        async def actor(ctx: Context) -> None:
            res = await reserve_memory(
                ctx,
                2048,
                net_memory_delta=0,
                mem_type=MemoryType.DEVICE,
                allow_overbooking=None,
            )
            assert res.mem_type == MemoryType.DEVICE
            assert res.size == 2048

        run_actor_network(
            actors=[actor(context)],
            py_executor=py_executor,
        )

    # Option disabled, should fail when no progress is possible.
    with make_context(
        dev_limit=1024,
        overwrite_options={
            "memory_reserve_timeout": "1ms",
            "allow_overbooking_by_default": "false",
        },
    ) as context:

        @define_actor()
        async def actor(ctx: Context) -> None:
            with pytest.raises(RuntimeError):
                await reserve_memory(
                    ctx,
                    2048,
                    net_memory_delta=0,
                    mem_type=MemoryType.DEVICE,
                    allow_overbooking=None,
                )

        run_actor_network(
            actors=[actor(context)],
            py_executor=py_executor,
        )
