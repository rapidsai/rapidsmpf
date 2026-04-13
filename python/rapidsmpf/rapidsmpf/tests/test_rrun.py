# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import multiprocessing
import os
import traceback
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from rapidsmpf.rrun import bind


def _run_in_subprocess(target: Callable[[], None]) -> None:
    """Execute ``target()`` in a forked child process.

    Because each call forks a new child, process-wide side-effects
    (CPU affinity, NUMA policy, environment variables) never leak into the
    pytest process. Any exception raised by ``target`` is propagated back to
    the caller.
    """
    ctx = multiprocessing.get_context("fork")
    parent_conn, child_conn = ctx.Pipe()

    def _wrapper() -> None:
        try:
            target()
            child_conn.send(None)
        except BaseException as exc:
            try:
                child_conn.send(exc)
            except Exception:
                child_conn.send(
                    RuntimeError(
                        f"{type(exc).__name__}: {exc}\n"
                        f"{''.join(traceback.format_tb(exc.__traceback__))}"
                    )
                )
        finally:
            child_conn.close()

    proc = ctx.Process(target=_wrapper)
    proc.start()
    proc.join(timeout=30)

    if proc.is_alive():
        proc.kill()
        proc.join()
        raise RuntimeError("Subprocess timed out after 30 seconds")

    if parent_conn.poll():
        exc = parent_conn.recv()
        if exc is not None:
            raise exc

    if proc.exitcode != 0:
        raise RuntimeError(f"Subprocess exited with code {proc.exitcode}")


class TestBindResolution:
    """GPU-ID resolution tests."""

    def test_explicit_gpu_id(self) -> None:
        def body() -> None:
            bind(gpu_id=0, cpu=False, memory=False, network=False)

        _run_in_subprocess(body)

    def test_cuda_visible_devices_fallback(self) -> None:
        def body() -> None:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            bind(cpu=False, memory=False, network=False)

        _run_in_subprocess(body)

    def test_no_gpu_id_raises(self) -> None:
        def body() -> None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            with pytest.raises(RuntimeError, match="no GPU ID specified"):
                bind(cpu=False, memory=False, network=False)

        _run_in_subprocess(body)

    def test_negative_gpu_id_raises(self) -> None:
        def body() -> None:
            with pytest.raises(ValueError, match="non-negative integer"):
                bind(gpu_id=-1, cpu=False, memory=False, network=False)

        _run_in_subprocess(body)

    def test_non_integer_gpu_id_raises(self) -> None:
        def body() -> None:
            with pytest.raises(ValueError, match="non-negative integer"):
                bind(gpu_id="0", cpu=False, memory=False, network=False)  # type: ignore[arg-type]

        _run_in_subprocess(body)

    def test_invalid_cuda_visible_devices_raises(self) -> None:
        def body() -> None:
            os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-abcdef12-3456"
            with pytest.raises(RuntimeError, match="not a valid GPU ID"):
                bind(cpu=False, memory=False, network=False)

        _run_in_subprocess(body)


class TestBindEffect:
    """Verify that bind() actually applies resource bindings.

    Because ``cucascade::memory::topology_discovery`` is not currently exposed
    to Python, the expected binding values (CPU cores, NUMA nodes, network
    devices) cannot be obtained from Python directly.  Tests here therefore
    perform basic smoke checks -- verifying that the call succeeds and that
    observable process state changes in the expected direction -- rather than
    asserting exact topology-derived values.
    """

    def test_cpu_affinity_changes(self) -> None:
        def body() -> None:
            before = os.sched_getaffinity(0)
            bind(gpu_id=0, cpu=True, memory=False, network=False)
            after = os.sched_getaffinity(0)
            if len(before) > 1:
                assert len(after) <= len(before), (
                    f"Expected affinity to narrow; before={before}, after={after}"
                )

        _run_in_subprocess(body)

    def test_cpu_binding_skipped_when_disabled(self) -> None:
        def body() -> None:
            before = os.sched_getaffinity(0)
            bind(gpu_id=0, cpu=False, memory=False, network=False)
            after = os.sched_getaffinity(0)
            assert before == after

        _run_in_subprocess(body)

    def test_memory_binding_succeeds(self) -> None:
        def body() -> None:
            bind(gpu_id=0, cpu=False, memory=True, network=False)

        _run_in_subprocess(body)

    def test_memory_binding_skipped_when_disabled(self) -> None:
        def body() -> None:
            bind(gpu_id=0, cpu=False, memory=False, network=False)

        _run_in_subprocess(body)

    def test_network_binding_sets_ucx_net_devices(self) -> None:
        def body() -> None:
            os.environ.pop("UCX_NET_DEVICES", None)
            bind(gpu_id=0, cpu=False, memory=False, network=True)
            val = os.environ.get("UCX_NET_DEVICES")
            # On systems without topology-adjacent NICs (e.g. CI without
            # mlx5 devices) bind() legitimately leaves UCX_NET_DEVICES
            # unset.  We only assert the value is non-empty when present.
            if val is not None:
                assert len(val) > 0

        _run_in_subprocess(body)

    def test_network_binding_skipped_when_disabled(self) -> None:
        def body() -> None:
            os.environ.pop("UCX_NET_DEVICES", None)
            bind(gpu_id=0, cpu=False, memory=False, network=False)
            assert os.environ.get("UCX_NET_DEVICES") is None

        _run_in_subprocess(body)
