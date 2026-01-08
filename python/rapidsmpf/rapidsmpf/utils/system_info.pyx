# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint64_t


cdef extern from "<rapidsmpf/system_info.hpp>" nogil:
    cdef uint64_t cpp_get_total_host_memory \
        "rapidsmpf::get_total_host_memory"() noexcept

    cdef uint64_t cpp_get_current_numa_node \
        "rapidsmpf::get_current_numa_node"() noexcept

    cdef uint64_t cpp_get_numa_node_host_memory \
        "rapidsmpf::get_numa_node_host_memory"(int numa_id) noexcept


def get_total_host_memory():
    """
    Get the total amount of system memory.

    Returns
    -------
    Total host memory in bytes.

    Notes
    -----
    On WSL and in containerized environments, the returned value reflects
    the memory visible to the Linux kernel instance, which may differ from
    the physical memory of the host.
    """
    return cpp_get_total_host_memory()


def get_current_numa_node():
    """
    Get the NUMA node ID associated with the calling thread.

    A NUMA (Non-Uniform Memory Access) node represents a group of CPU cores
    and memory that have faster access to each other than to memory attached
    to other nodes. On NUMA systems, binding allocations and threads to the
    same NUMA node can reduce memory access latency and improve bandwidth.

    The returned value corresponds to the NUMA node on which the calling
    thread is currently executing. This value may change if the thread
    migrates between CPUs.

    If NUMA support is not available or cannot be queried, this function
    returns 0, corresponding to the single implicit NUMA node on non-NUMA
    systems.

    Returns
    -------
    NUMA node ID of the calling thread, or 0 if NUMA is unavailable.
    """
    return cpp_get_current_numa_node()


def get_numa_node_host_memory(numa_id = None):
    """
    Get the total amount of host memory for a NUMA node.

    Parameters
    ----------
    numa_id
        NUMA node for which to query the total host memory. If not provided,
        the NUMA node of the calling thread is used.

    Returns
    -------
    Total host memory of the NUMA node in bytes.

    Notes
    -----
    If NUMA support is not available or the node size cannot be determined,
    this function falls back to returning the total host memory.
    """
    cdef int _numa_id
    if numa_id is None:
        _numa_id = cpp_get_current_numa_node()
    else:
        _numa_id = numa_id
    return cpp_get_numa_node_host_memory(_numa_id)
