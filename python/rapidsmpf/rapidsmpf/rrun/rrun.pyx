# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import os

from libc.stdlib cimport getenv as c_getenv
from libcpp cimport bool as cbool
from libcpp.optional cimport optional

from rapidsmpf._detail.exception_handling cimport ex_handler


cdef extern from "<rrun/rrun.hpp>" namespace "rapidsmpf::rrun" nogil:
    cdef cppclass cpp_bind_options "rapidsmpf::rrun::bind_options":
        cbool cpu
        cbool memory
        cbool network
        cbool verbose

    void cpp_bind "rapidsmpf::rrun::bind"(
        optional[unsigned int] gpu_id,
        const cpp_bind_options& options,
    ) except +ex_handler


# Environment variables that bind() may modify via C setenv().
_ENV_VARS_MODIFIED_BY_BIND = ("UCX_NET_DEVICES", "CUDA_VISIBLE_DEVICES")


cdef _sync_c_env_to_python():
    """Propagate C-level environment changes made by bind() into os.environ.

    The C++ bind() implementation uses setenv()/unsetenv() which modify the C
    environment directly.  Python's os.environ is a cached dict that is only
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


def bind(
    gpu_id=None,
    *,
    cpu=True,
    memory=True,
    network=True,
    verbose=False,
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
    verbose
        Print warnings to stderr on binding failures (default ``False``).

    Raises
    ------
    RuntimeError
        If no GPU ID can be determined or the resolved GPU is not found
        in the discovered topology.
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
    opts.verbose = verbose

    with nogil:
        cpp_bind(c_gpu_id, opts)
    _sync_c_env_to_python()
