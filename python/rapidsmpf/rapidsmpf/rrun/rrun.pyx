# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

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

    GPU resolution order:

    1. Use *gpu_id* if provided.
    2. Otherwise, parse the first entry of the ``CUDA_VISIBLE_DEVICES``
       environment variable.
    3. If neither is available, raise :class:`RuntimeError`.

    Parameters
    ----------
    gpu_id : int or None
        Physical GPU device index (as reported by ``nvidia-smi``).
        When ``None``, the first GPU in ``CUDA_VISIBLE_DEVICES`` is used.
    cpu : bool
        Set CPU affinity to cores near the GPU (default ``True``).
    memory : bool
        Set NUMA memory policy to nodes near the GPU (default ``True``).
    network : bool
        Set ``UCX_NET_DEVICES`` to NICs near the GPU (default ``True``).
    verbose : bool
        Print warnings to stderr on binding failures (default ``False``).

    Raises
    ------
    RuntimeError
        If no GPU ID can be determined or the resolved GPU is not found
        in the discovered topology.
    """
    cdef optional[unsigned int] c_gpu_id
    if gpu_id is not None:
        c_gpu_id = <unsigned int>gpu_id

    cdef cpp_bind_options opts
    opts.cpu = cpu
    opts.memory = memory
    opts.network = network
    opts.verbose = verbose

    with nogil:
        cpp_bind(c_gpu_id, opts)
