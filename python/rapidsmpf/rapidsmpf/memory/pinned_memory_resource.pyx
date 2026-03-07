# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libcpp.memory cimport make_shared
from libcpp.optional cimport make_optional

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.config cimport Options, cpp_Options
from rapidsmpf.utils.system_info import get_current_numa_node


cdef extern from "<rapidsmpf/memory/pinned_memory_resource.hpp>" nogil:
    cdef bool_t cpp_is_pinned_memory_resources_supported \
        "rapidsmpf::is_pinned_memory_resources_supported"(...) except +ex_handler

    cdef shared_ptr[cpp_PinnedMemoryResource] cpp_from_options \
        "rapidsmpf::PinnedMemoryResource::from_options"(
            cpp_Options options
        ) except +ex_handler


class PinnedPoolProperties:
    """
    Properties for configuring a pinned memory pool.

    Parameters
    ----------
    initial_pool_size
        Initial size of the pool in bytes. A larger initial size can improve
        performance for the first allocation. Defaults to 0.
    max_pool_size
        Maximum size of the pool in bytes. ``None`` means no limit.
    """
    def __init__(self, initial_pool_size: int = 0, max_pool_size=None):
        self.initial_pool_size = initial_pool_size
        self.max_pool_size = max_pool_size


cpdef bool_t is_pinned_memory_resources_supported():
    """
    Check whether pinned memory resources are supported for the current CUDA version.

    RapidsMPF requires CUDA 12.6 or newer to support pinned memory resources.
    """
    cdef bool_t ret
    with nogil:
        ret = cpp_is_pinned_memory_resources_supported()
    return ret


cdef class PinnedMemoryResource:
    """
    Memory resource that provides pinned (page-locked) host memory using a pool.

    The resource allocates and deallocates pinned host memory asynchronously
    through CUDA streams. Pinned memory enables higher bandwidth and lower
    latency for device transfers compared to regular pageable host memory.

    The pool has no maximum size. To limit its growth, use
    ``LimitAvailableMemory`` or a similar mechanism.

    Parameters
    ----------
    numa_id
        NUMA node from which memory should be allocated. By default, the
        resource uses the NUMA node of the calling thread.
    pool_properties
        Properties for configuring the pinned memory pool. If ``None``,
        default pool properties are used.

    Raises
    ------
    RuntimeError
        If pinned host memory pools are not supported by the current CUDA
        version.
    """
    def __init__(self, numa_id=None, pool_properties=None):
        cdef cpp_PinnedPoolProperties props
        if pool_properties is not None:
            props.initial_pool_size = pool_properties.initial_pool_size
            if pool_properties.max_pool_size is not None:
                props.max_pool_size = make_optional[size_t](
                    <size_t?>pool_properties.max_pool_size
                )
        numa_id = numa_id if numa_id is not None else get_current_numa_node()
        self._handle = make_shared[cpp_PinnedMemoryResource](
            <int?>(numa_id), props
        )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @staticmethod
    def make_if_available(numa_id=None, pool_properties=None):
        """
        Create a pinned memory resource if the system supports pinned memory.

        Parameters
        ----------
        numa_id
            NUMA node to associate with the resource. Defaults to the current
            NUMA node.
        pool_properties
            Properties for configuring the pinned memory pool. If ``None``,
            default pool properties are used.

        Returns
        -------
        A pinned memory resource when supported, otherwise None.
        """
        if is_pinned_memory_resources_supported():
            return PinnedMemoryResource(numa_id, pool_properties)
        return None

    @classmethod
    def from_options(cls, Options options not None):
        """
        Construct from configuration options.

        Parameters
        ----------
        options
            Configuration options.

        Returns
        -------
        The constructed PinnedMemoryResource instance if pinned memory is enabled
        and supported by the system, otherwise ``None``.
        """
        cdef shared_ptr[cpp_PinnedMemoryResource] handle
        with nogil:
            handle = cpp_from_options(options._handle)
        if not handle:
            return None
        cdef PinnedMemoryResource ret = cls.__new__(cls)
        ret._handle = handle
        return ret
