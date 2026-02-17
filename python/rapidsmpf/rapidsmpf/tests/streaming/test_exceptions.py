# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rapidsmpf.streaming.core.leaf_node import pull_from_channel
from rapidsmpf.streaming.core.node import define_py_actor, run_actor_graph

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.core.message import Payload
    from rapidsmpf.streaming.core.node import CppActor, PyActor


@define_py_actor()
async def task_that_throws(ctx: Context, ch_in: Channel, ch_out: Channel) -> None:
    raise RuntimeError("Throwing in task")


@define_py_actor()
async def task_that_spins(ctx: Context, ch_in: Channel) -> None:
    while await ch_in.recv(ctx) is not None:
        pass


def test_task_exceptions(context: Context, py_executor: ThreadPoolExecutor) -> None:
    ch1: Channel[Payload] = context.create_channel()
    ch2: Channel[Payload] = context.create_channel()
    ch3: Channel[Payload] = context.create_channel()

    pull_task, deferred = pull_from_channel(context, ch3)

    nodes: list[CppActor | PyActor] = [
        task_that_throws(context, ch1, ch2),
        task_that_spins(context, ch3),
        pull_task,
    ]

    with pytest.raises(RuntimeError, match="Throwing in task"):
        run_actor_graph(nodes=nodes, py_executor=py_executor)

    messages = deferred.release()
    assert len(messages) == 0
