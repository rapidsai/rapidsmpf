# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport make_shared

from rapidsmpf.config cimport Options, cpp_Options


cdef extern from "<rapidsmpf/memory/pinned_memory_resource.hpp>" nogil:
    cdef bool_t cpp_is_pinned_memory_resources_supported \
        "rapidsmpf::is_pinned_memory_resources_supported"(...) except +

    cdef shared_ptr[cpp_PinnedMemoryResource] cpp_from_options \
        "rapidsmpf::PinnedMemoryResource::from_options"(cpp_Options options) except +


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

    Raises
    ------
    RuntimeError
        If pinned host memory pools are not supported by the current CUDA
        version.
    """
    def __init__(self, numa_id = None):
        if numa_id is None:
            self._handle = make_shared[cpp_PinnedMemoryResource]()
        else:
            self._handle = make_shared[cpp_PinnedMemoryResource](<int?>numa_id)

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @staticmethod
    def make_if_available(numa_id = None):
        """
        Create a pinned memory resource if the system supports pinned memory.

        Parameters
        ----------
        numa_id
            NUMA node to associate with the resource. Defaults to the current
            NUMA node.

        Returns
        -------
        A pinned memory resource when supported, otherwise None.
        """
        if is_pinned_memory_resources_supported():
            return PinnedMemoryResource(numa_id)
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
