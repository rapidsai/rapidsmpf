# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Concurrency smoke test for the streaming actor network.

Runs 2 Python delay actors concurrently in a
single ``run_actor_network`` call. Each actor sleeps for ``sleep_seconds``
and is wrapped in an NVTX range so the layout is visible under Nsight
Systems.

The Context is configured with 4 streaming (libcoro) threads so the two
C++ actors get one worker each, and the Python executor is given 4
workers (note that ``run_actor_network`` only uses one of them to host the
asyncio loop -- Python-side concurrency comes from ``asyncio``).

With true concurrency, total wall time should be close to ``sleep_seconds``
rather than ``2 * sleep_seconds``.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import rmm.mr

from rapidsmpf.communicator.single import (
    new_communicator as single_process_comm,
)
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.streaming.core.actor import (
    define_actor,
    run_actor_network,
)
from rapidsmpf.streaming.core.context import Context


def test_mixed_actor_concurrency() -> None:
    """2 Python delay actors should run concurrently, not serially."""
    nvtx = pytest.importorskip("nvtx")

    sleep_seconds = 1.0

    options = Options({"num_streaming_threads": "4"})
    options.insert_if_absent(get_environment_variables())

    comm = single_process_comm(options, ProgressThread())
    br = BufferResource(
        RmmResourceAdaptor(rmm.mr.get_current_device_resource())
    )

    py_executor = ThreadPoolExecutor(max_workers=4)

    with Context(comm.logger, br, options) as ctx:
        assert int(options.get_strings()["num_streaming_threads"]) == 4

        @define_actor()
        async def py_delay_actor(ctx: Context, /, *, name: str) -> None:
            # Foreground (blocking) sleep: holds the GIL and blocks the
            # asyncio event loop, so two Python actors using this body will
            # serialize on the loop's single thread.
            with nvtx.annotate(message=name, color="green"):
                time.sleep(sleep_seconds)

        actors = [
            py_delay_actor(ctx, name="py_actor_0"),
            py_delay_actor(ctx, name="py_actor_1"),
        ]

        with nvtx.annotate(message="run_actor_network", color="blue"):
            start = time.monotonic()
            run_actor_network(
                actors=actors,
                py_executor=py_executor,
            )
            elapsed = time.monotonic() - start

    py_executor.shutdown(wait=True)

    # If all four actors ran serially we'd see ~4 * sleep_seconds. With true
    # concurrency we expect roughly sleep_seconds plus a small overhead.
    # Use a generous bound to avoid flakiness on busy CI runners.
    assert elapsed < 1.5 * sleep_seconds, (
        f"Actors appear to have run serially: "
        f"elapsed={elapsed:.2f}s, expected ~{sleep_seconds:.2f}s"
    )
