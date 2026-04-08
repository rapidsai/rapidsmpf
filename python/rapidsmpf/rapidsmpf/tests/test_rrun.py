# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ctypes
import os

import pytest

from rapidsmpf.rrun import bind

_libc = ctypes.CDLL(None, use_errno=True)
_libc.getenv.restype = ctypes.c_char_p
_libc.getenv.argtypes = [ctypes.c_char_p]


def _c_getenv(name: str) -> str | None:
    """Read an environment variable via C getenv().

    Python's os.environ is a cached dict that does not reflect changes made by
    C-level setenv() (e.g. from the C++ bind() implementation).
    """
    val = _libc.getenv(name.encode())
    return val.decode() if val is not None else None


class TestBindResolution:
    """GPU-ID resolution tests (no real binding side-effects)."""

    def test_explicit_gpu_id(self) -> None:
        bind(gpu_id=0, cpu=False, memory=False, network=False)

    def test_cuda_visible_devices_fallback(self) -> None:
        old = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            bind(cpu=False, memory=False, network=False)
        finally:
            if old is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old

    def test_no_gpu_id_raises(self) -> None:
        old = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            with pytest.raises(RuntimeError, match="no GPU ID specified"):
                bind(cpu=False, memory=False, network=False)
        finally:
            if old is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = old

    def test_invalid_cuda_visible_devices_raises(self) -> None:
        old = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-abcdef12-3456"
            with pytest.raises(RuntimeError, match="not a valid GPU ID"):
                bind(cpu=False, memory=False, network=False)
        finally:
            if old is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old


class TestBindEffect:
    """Verify that bind() actually applies resource bindings."""

    def test_cpu_affinity_changes(self) -> None:
        before = os.sched_getaffinity(0)
        bind(gpu_id=0, cpu=True, memory=False, network=False)
        after = os.sched_getaffinity(0)
        # Binding should restrict the affinity (unless the system has only
        # one NUMA domain and all cores are already assigned to GPU 0).
        if len(before) > 1:
            assert len(after) <= len(before), (
                f"Expected affinity to narrow; before={before}, after={after}"
            )
        os.sched_setaffinity(0, before)

    def test_cpu_binding_skipped_when_disabled(self) -> None:
        before = os.sched_getaffinity(0)
        bind(gpu_id=0, cpu=False, memory=False, network=False)
        after = os.sched_getaffinity(0)
        assert before == after

    def test_network_binding_sets_ucx_net_devices(self) -> None:
        old = _c_getenv("UCX_NET_DEVICES")
        try:
            bind(gpu_id=0, cpu=False, memory=False, network=True)
            val = _c_getenv("UCX_NET_DEVICES")
            assert val is not None
            assert len(val) > 0
        finally:
            if old is None:
                os.unsetenv("UCX_NET_DEVICES")
            else:
                os.putenv("UCX_NET_DEVICES", old)

    def test_network_binding_skipped_when_disabled(self) -> None:
        old = _c_getenv("UCX_NET_DEVICES")
        os.unsetenv("UCX_NET_DEVICES")
        try:
            bind(gpu_id=0, cpu=False, memory=False, network=False)
            assert _c_getenv("UCX_NET_DEVICES") is None
        finally:
            if old is not None:
                os.putenv("UCX_NET_DEVICES", old)
