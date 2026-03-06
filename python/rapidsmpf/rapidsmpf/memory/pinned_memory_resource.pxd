# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libcpp cimport bool as bool_t
from libcpp.memory cimport shared_ptr
from libcpp.optional cimport optional

from rapidsmpf._detail.exception_handling cimport ex_handler


cdef extern from "<rapidsmpf/memory/pinned_memory_resource.hpp>" nogil:
    cdef cppclass cpp_PinnedPoolProperties "rapidsmpf::PinnedPoolProperties":
        size_t initial_pool_size
        optional[size_t] max_pool_size

    cdef cppclass cpp_PinnedMemoryResource"rapidsmpf::PinnedMemoryResource":
        cpp_PinnedMemoryResource() except +ex_handler
        cpp_PinnedMemoryResource(int numa_id) except +ex_handler
        cpp_PinnedMemoryResource(
            int numa_id, cpp_PinnedPoolProperties pool_properties
        ) except +ex_handler

cpdef bool_t is_pinned_memory_resources_supported()

cdef class PinnedMemoryResource:
    cdef shared_ptr[cpp_PinnedMemoryResource] _handle
