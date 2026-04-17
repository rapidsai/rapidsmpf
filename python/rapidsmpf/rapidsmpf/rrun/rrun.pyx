# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os

from libc.stdlib cimport getenv as c_getenv
from libcpp cimport bool as cbool
from libcpp.optional cimport optional
from libcpp.string cimport string as cppstring
from libcpp.vector cimport vector

from rapidsmpf._detail.exception_handling cimport ex_handler


cdef extern from "<rrun/rrun.hpp>" namespace "rapidsmpf::rrun" nogil:
    cdef cppclass cpp_bind_options "rapidsmpf::rrun::bind_options":
        cbool cpu
        cbool memory
        cbool network
        cbool verify

    cdef cppclass cpp_resource_binding "rapidsmpf::rrun::resource_binding":
        int rank
        int gpu_id
        cppstring gpu_pci_bus_id
        cppstring cpu_affinity
        vector[int] numa_nodes
        cppstring ucx_net_devices

    cdef cppclass cpp_expected_binding "rapidsmpf::rrun::expected_binding":
        cppstring cpu_affinity
        vector[int] memory_binding
        vector[cppstring] network_devices

    cdef cppclass cpp_binding_validation "rapidsmpf::rrun::binding_validation":
        cbool cpu_ok
        cbool numa_ok
        cbool ucx_ok
        cppstring expected_ucx_devices
        cbool all_passed()

    void cpp_bind "rapidsmpf::rrun::bind"(
        optional[unsigned int] gpu_id,
        const cpp_bind_options& options,
    ) except +ex_handler

    cpp_resource_binding cpp_check_binding "rapidsmpf::rrun::check_binding"(
        int gpu_id_hint,
    ) except +ex_handler

    cpp_binding_validation cpp_validate_binding "rapidsmpf::rrun::validate_binding"(
        const cpp_resource_binding& actual,
        const cpp_expected_binding& expected,
    ) except +ex_handler


# Environment variables that bind() may modify via C setenv().
_ENV_VARS_MODIFIED_BY_BIND = ("UCX_NET_DEVICES", "CUDA_VISIBLE_DEVICES")


cdef _sync_c_env_to_python():
    """Propagate C-level environment changes made by bind() into os.environ.

    The C++ bind() implementation uses setenv()/unsetenv() which modify the C
    environment directly. Python's os.environ is a cached dict that is only
    updated when Python code writes to it, so we must manually synchronize the
    variables that bind() may have touched.
    """
    cdef const char* val
    for name in _ENV_VARS_MODIFIED_BY_BIND:
        val = c_getenv(name.encode())
        if val != NULL:
            os.environ[name] = val.decode()
        else:
            os.environ.pop(name, None)


@dataclasses.dataclass(frozen=True)
class ResourceBinding:
    """
    Live resource binding configuration collected from the running process.

    Holds the CPU affinity, NUMA memory binding, and network device
    configuration that are currently in effect.

    Parameters
    ----------
    rank
        Process rank, or ``None`` if not available.
    gpu_id
        GPU device ID, or ``None`` if not available.
    gpu_pci_bus_id
        GPU PCI bus ID (empty if unavailable).
    cpu_affinity
        CPU affinity string (e.g., ``"0-19,40-59"``).
    numa_nodes
        NUMA node IDs bound to this process.
    ucx_net_devices
        Value of the ``UCX_NET_DEVICES`` environment variable.

    See Also
    --------
    check_binding : Collect the live binding of the calling process.
    """

    rank: int | None
    gpu_id: int | None
    gpu_pci_bus_id: str
    cpu_affinity: str
    numa_nodes: list[int]
    ucx_net_devices: str


@dataclasses.dataclass(frozen=True)
class ExpectedBinding:
    """
    Expected resource binding derived from topology information.

    Represents the binding configuration that *should* be in effect for a
    given GPU according to the system topology.

    Parameters
    ----------
    cpu_affinity
        Expected CPU affinity list.
    memory_binding
        Expected NUMA node IDs.
    network_devices
        Expected network devices.

    See Also
    --------
    validate_binding : Compare actual vs. expected bindings.
    """

    cpu_affinity: str = ""
    memory_binding: list[int] = dataclasses.field(default_factory=list)
    network_devices: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class BindingValidation:
    """
    Results of validating actual vs. expected resource bindings.

    Parameters
    ----------
    cpu_ok
        CPU affinity check passed.
    numa_ok
        NUMA binding check passed.
    ucx_ok
        UCX network devices check passed.
    expected_ucx_devices
        Expected UCX devices as a comma-separated string.

    See Also
    --------
    validate_binding : Produce a validation result.
    """

    cpu_ok: bool
    numa_ok: bool
    ucx_ok: bool
    expected_ucx_devices: str

    def all_passed(self) -> bool:
        """Return ``True`` if all validation checks passed."""
        return self.cpu_ok and self.numa_ok and self.ucx_ok


def check_binding(gpu_id_hint=None):
    """
    Collect the live resource binding of the calling process.

    Queries the current CPU affinity, NUMA memory nodes, UCX network
    device configuration, process rank, and GPU information. Fields that
    cannot be determined (e.g. rank when no launcher environment is set,
    or GPU ID when ``CUDA_VISIBLE_DEVICES`` is absent and no hint is
    given) are returned as ``None``.

    Parameters
    ----------
    gpu_id_hint
        GPU device index hint. When a non-negative integer the value is
        stored directly; when ``None`` the GPU ID is read from
        ``CUDA_VISIBLE_DEVICES``. When a valid GPU ID is available the
        PCI bus ID is also queried.

    Returns
    -------
    ResourceBinding
        The collected resource binding.
    """
    cdef int c_hint = gpu_id_hint if gpu_id_hint is not None else -1
    cdef cpp_resource_binding c_result
    with nogil:
        c_result = cpp_check_binding(c_hint)
    return ResourceBinding(
        rank=c_result.rank if c_result.rank >= 0 else None,
        gpu_id=c_result.gpu_id if c_result.gpu_id >= 0 else None,
        gpu_pci_bus_id=c_result.gpu_pci_bus_id.decode(),
        cpu_affinity=c_result.cpu_affinity.decode(),
        numa_nodes=list(c_result.numa_nodes),
        ucx_net_devices=c_result.ucx_net_devices.decode(),
    )


def validate_binding(actual, expected):
    """
    Validate an actual resource binding against an expected one.

    Compares the live *actual* binding with *expected* and reports
    per-resource pass/fail status.

    Parameters
    ----------
    actual : ResourceBinding
        Live resource binding (from :func:`check_binding`).
    expected : ExpectedBinding
        Expected binding (from topology or another source).

    Returns
    -------
    BindingValidation
        Per-resource validation results.
    """
    cdef cpp_resource_binding c_actual
    c_actual.rank = actual.rank if actual.rank is not None else -1
    c_actual.gpu_id = actual.gpu_id if actual.gpu_id is not None else -1
    c_actual.gpu_pci_bus_id = actual.gpu_pci_bus_id.encode()
    c_actual.cpu_affinity = actual.cpu_affinity.encode()
    c_actual.numa_nodes = actual.numa_nodes
    c_actual.ucx_net_devices = actual.ucx_net_devices.encode()

    cdef cpp_expected_binding c_expected
    c_expected.cpu_affinity = expected.cpu_affinity.encode()
    c_expected.memory_binding = expected.memory_binding
    cdef vector[cppstring] c_net_devs
    for dev in expected.network_devices:
        c_net_devs.push_back((<str>dev).encode())
    c_expected.network_devices = c_net_devs

    cdef cpp_binding_validation c_result
    with nogil:
        c_result = cpp_validate_binding(c_actual, c_expected)
    return BindingValidation(
        cpu_ok=c_result.cpu_ok,
        numa_ok=c_result.numa_ok,
        ucx_ok=c_result.ucx_ok,
        expected_ucx_devices=c_result.expected_ucx_devices.decode(),
    )


def bind(
    gpu_id=None,
    *,
    cpu=True,
    memory=True,
    network=True,
    verify=True,
):
    """
    Bind the calling process to resources topologically close to a GPU.

    Discovers the system topology, then applies CPU affinity, NUMA memory
    binding, and/or network device configuration as requested.

    .. warning::
        This function is **not thread-safe**. It temporarily modifies the
        ``CUDA_VISIBLE_DEVICES`` environment variable during topology
        discovery and mutates process-wide state (CPU affinity, NUMA memory
        policy, and the ``UCX_NET_DEVICES`` environment variable). It should
        be called exactly once per process, ideally early in initialization
        and before other threads are spawned.

    GPU resolution order:

    1. Use ``gpu_id`` if provided.
    2. Otherwise, parse the first entry of the ``CUDA_VISIBLE_DEVICES``
       environment variable.
    3. If neither is available, raise :class:`RuntimeError`.

    Parameters
    ----------
    gpu_id
        Physical GPU device index (as reported by ``nvidia-smi``).
        When ``None``, the first GPU in ``CUDA_VISIBLE_DEVICES`` is used.
    cpu
        Set CPU affinity to cores near the GPU (default ``True``).
    memory
        Set NUMA memory policy to nodes near the GPU (default ``True``).
    network
        Set ``UCX_NET_DEVICES`` to NICs near the GPU (default ``True``).
    verify
        Read back and verify that bindings match the requested
        configuration after applying them (default ``True``).

    Raises
    ------
    RuntimeError
        If no GPU ID can be determined, topology discovery fails, the
        resolved GPU is not found in the discovered topology, an
        enabled binding (CPU affinity, NUMA memory policy, network
        devices) could not be applied, or post-bind verification
        detects a mismatch between the requested and actual state.
    ValueError
        If ``gpu_id`` is not a non-negative integer.
    """
    cdef optional[unsigned int] c_gpu_id
    if gpu_id is not None:
        if not isinstance(gpu_id, int) or gpu_id < 0:
            raise ValueError(
                f"gpu_id must be a non-negative integer, got {gpu_id!r}"
            )
        c_gpu_id = <unsigned int>gpu_id

    cdef cpp_bind_options opts
    opts.cpu = cpu
    opts.memory = memory
    opts.network = network
    opts.verify = verify

    with nogil:
        cpp_bind(c_gpu_id, opts)
    _sync_c_env_to_python()
