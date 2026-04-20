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

from rapidsmpf.rrun import (
    BindingValidation,
    ExpectedBinding,
    ResourceBinding,
    bind,
    check_binding,
    validate_binding,
)

_LAUNCHER_VARS = ("RRUN_RANK", "PMIX_RANK", "SLURM_PROCID")

_has_launcher = any(os.environ.get(v) is not None for v in _LAUNCHER_VARS)

requires_launcher = pytest.mark.skipif(
    not _has_launcher,
    reason="Requires a process launcher (rrun/mpirun/srun)",
)

requires_no_launcher = pytest.mark.skipif(
    _has_launcher,
    reason="Must run without a process launcher",
)


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


class TestCheckBinding:
    """Tests for check_binding() that work in any environment."""

    def test_returns_resource_binding(self) -> None:
        def body() -> None:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            result = check_binding()
            assert isinstance(result, ResourceBinding)
            assert result.gpu_id == 0
            assert isinstance(result.cpu_affinity, str)
            assert isinstance(result.numa_nodes, list)
            assert isinstance(result.ucx_net_devices, str)

        _run_in_subprocess(body)

    def test_explicit_gpu_id_hint(self) -> None:
        def body() -> None:
            result = check_binding(gpu_id_hint=42)
            assert result.gpu_id == 42

        _run_in_subprocess(body)

    def test_none_hint_falls_back_to_cvd(self) -> None:
        def body() -> None:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            result = check_binding(gpu_id_hint=None)
            assert result.gpu_id == 0

        _run_in_subprocess(body)

    def test_default_hint_uses_cvd(self) -> None:
        def body() -> None:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            result = check_binding()
            assert result.gpu_id == 0

        _run_in_subprocess(body)

    def test_cpu_affinity_is_nonempty(self) -> None:
        def body() -> None:
            result = check_binding(gpu_id_hint=0)
            assert len(result.cpu_affinity) > 0

        _run_in_subprocess(body)


@requires_launcher
class TestCheckBindingWithLauncher:
    """Tests for check_binding() when a process launcher is present."""

    def test_rank_is_populated(self) -> None:
        def body() -> None:
            result = check_binding(gpu_id_hint=0)
            assert result.rank is not None
            assert result.rank >= 0

        _run_in_subprocess(body)

    @pytest.mark.skipif(
        os.environ.get("CUDA_VISIBLE_DEVICES") is None,
        reason="CUDA_VISIBLE_DEVICES not set by launcher",
    )
    def test_gpu_id_from_launcher(self) -> None:
        def body() -> None:
            result = check_binding()
            assert result.gpu_id is not None
            assert result.gpu_id >= 0

        _run_in_subprocess(body)


@requires_no_launcher
class TestCheckBindingWithoutLauncher:
    """Tests for check_binding() when no process launcher is present."""

    def test_rank_is_none(self) -> None:
        def body() -> None:
            result = check_binding(gpu_id_hint=0)
            assert result.rank is None

        _run_in_subprocess(body)

    def test_gpu_id_is_none_without_hint_or_cvd(self) -> None:
        def body() -> None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            result = check_binding()
            assert result.gpu_id is None

        _run_in_subprocess(body)


class TestValidateBinding:
    """Tests for validate_binding() with synthetic data (no GPU needed)."""

    def test_all_pass_when_matching(self) -> None:
        actual = ResourceBinding(
            rank=None,
            gpu_id=0,
            gpu_pci_bus_id="",
            cpu_affinity="0-3",
            numa_nodes=[0],
            ucx_net_devices="mlx5_0",
        )
        expected = ExpectedBinding(
            cpu_affinity="0-3",
            memory_binding=[0],
            network_devices=["mlx5_0"],
        )
        result = validate_binding(actual, expected)
        assert isinstance(result, BindingValidation)
        assert result.cpu_ok is True
        assert result.numa_ok is True
        assert result.ucx_ok is True
        assert result.all_passed() is True

    def test_cpu_mismatch_detected(self) -> None:
        actual = ResourceBinding(
            rank=None,
            gpu_id=0,
            gpu_pci_bus_id="",
            cpu_affinity="4-7",
            numa_nodes=[],
            ucx_net_devices="",
        )
        expected = ExpectedBinding(cpu_affinity="0-3")
        result = validate_binding(actual, expected)
        assert result.cpu_ok is False
        assert result.all_passed() is False

    def test_numa_mismatch_detected(self) -> None:
        actual = ResourceBinding(
            rank=None,
            gpu_id=0,
            gpu_pci_bus_id="",
            cpu_affinity="",
            numa_nodes=[1],
            ucx_net_devices="",
        )
        expected = ExpectedBinding(memory_binding=[0])
        result = validate_binding(actual, expected)
        assert result.numa_ok is False
        assert result.all_passed() is False

    def test_numa_passes_when_any_node_matches(self) -> None:
        actual = ResourceBinding(
            rank=None,
            gpu_id=0,
            gpu_pci_bus_id="",
            cpu_affinity="",
            numa_nodes=[1, 0],
            ucx_net_devices="",
        )
        expected = ExpectedBinding(memory_binding=[0])
        result = validate_binding(actual, expected)
        assert result.numa_ok is True

    def test_ucx_mismatch_detected(self) -> None:
        actual = ResourceBinding(
            rank=None,
            gpu_id=0,
            gpu_pci_bus_id="",
            cpu_affinity="",
            numa_nodes=[],
            ucx_net_devices="mlx5_1",
        )
        expected = ExpectedBinding(network_devices=["mlx5_0"])
        result = validate_binding(actual, expected)
        assert result.ucx_ok is False
        assert result.expected_ucx_devices == "mlx5_0"
        assert result.all_passed() is False

    def test_ucx_order_independent(self) -> None:
        actual = ResourceBinding(
            rank=None,
            gpu_id=0,
            gpu_pci_bus_id="",
            cpu_affinity="",
            numa_nodes=[],
            ucx_net_devices="mlx5_1,mlx5_0",
        )
        expected = ExpectedBinding(network_devices=["mlx5_0", "mlx5_1"])
        result = validate_binding(actual, expected)
        assert result.ucx_ok is True

    def test_empty_expected_is_all_pass(self) -> None:
        actual = ResourceBinding(
            rank=None,
            gpu_id=0,
            gpu_pci_bus_id="",
            cpu_affinity="0-7",
            numa_nodes=[0],
            ucx_net_devices="mlx5_0",
        )
        expected = ExpectedBinding()
        result = validate_binding(actual, expected)
        assert result.all_passed() is True


class TestDataclasses:
    """Tests for the dataclass types themselves."""

    def test_resource_binding_is_frozen(self) -> None:
        rb = ResourceBinding(
            rank=0,
            gpu_id=0,
            gpu_pci_bus_id="",
            cpu_affinity="0-3",
            numa_nodes=[0],
            ucx_net_devices="mlx5_0",
        )
        with pytest.raises(AttributeError):
            rb.rank = 1  # type: ignore[misc]

    def test_expected_binding_defaults(self) -> None:
        eb = ExpectedBinding()
        assert eb.cpu_affinity == ""
        assert eb.memory_binding == []
        assert eb.network_devices == []

    def test_binding_validation_all_passed(self) -> None:
        bv = BindingValidation(
            cpu_ok=True, numa_ok=True, ucx_ok=True, expected_ucx_devices=""
        )
        assert bv.all_passed() is True

    def test_binding_validation_not_all_passed(self) -> None:
        bv = BindingValidation(
            cpu_ok=True, numa_ok=False, ucx_ok=True, expected_ucx_devices=""
        )
        assert bv.all_passed() is False


class TestBindThenValidate:
    """Integration: bind() then check_binding()/validate_binding()."""

    def test_bind_and_check_produces_valid_result(self) -> None:
        def body() -> None:
            bind(gpu_id=0, cpu=True, memory=True, network=True, verify=True)
            result = check_binding(gpu_id_hint=0)
            assert result.gpu_id == 0
            assert len(result.cpu_affinity) > 0

        _run_in_subprocess(body)

    def test_bind_with_verify_false_succeeds(self) -> None:
        def body() -> None:
            bind(gpu_id=0, cpu=True, memory=True, network=True, verify=False)

        _run_in_subprocess(body)

    @requires_launcher
    def test_rank_populated_after_bind(self) -> None:
        def body() -> None:
            bind(gpu_id=0, cpu=True, memory=True, network=True)
            result = check_binding(gpu_id_hint=0)
            assert result.rank is not None
            assert result.rank >= 0

        _run_in_subprocess(body)

    @requires_no_launcher
    def test_rank_none_after_bind_without_launcher(self) -> None:
        def body() -> None:
            bind(gpu_id=0, cpu=True, memory=True, network=True)
            result = check_binding(gpu_id_hint=0)
            assert result.rank is None

        _run_in_subprocess(body)
