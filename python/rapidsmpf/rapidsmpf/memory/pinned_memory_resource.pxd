# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libcpp cimport bool as bool_t
from libcpp.optional cimport optional
from rmm.librmm.cuda_stream_view cimport cuda_stream_view

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.config cimport cpp_Options


cdef extern from "<rapidsmpf/memory/pinned_memory_resource.hpp>" nogil:
    cdef cppclass cpp_PinnedMemoryResource"rapidsmpf::PinnedMemoryResource":
        void* allocate(cuda_stream_view, size_t) except +ex_handler
        void deallocate(cuda_stream_view, void*, size_t)

    cdef cppclass cpp_PinnedPoolProperties"rapidsmpf::PinnedPoolProperties":
        size_t initial_pool_size
        optional[size_t] max_pool_size
        int numa_id

    optional[cpp_PinnedPoolProperties] pinned_pool_properties_from_options \
        "rapidsmpf::pinned_pool_properties_from_options"(
            cpp_Options options
        ) except +ex_handler

cpdef bool_t is_pinned_memory_resources_supported()

cdef object create_pinned_pool_properties_from_cpp(cpp_PinnedPoolProperties props)

cdef class PinnedMemoryResource:
    cdef optional[cpp_PinnedMemoryResource] _handle

    @staticmethod
    cdef PinnedMemoryResource from_handle(
        const optional[cpp_PinnedMemoryResource]& handle
    )
