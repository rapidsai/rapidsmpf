# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool as bool_t
from libcpp.optional cimport optional
from rmm.librmm.cuda_stream_view cimport cuda_stream_view

from rapidsmpf._detail.exception_handling cimport ex_handler


cdef extern from "<rapidsmpf/memory/pinned_memory_resource.hpp>" nogil:
    cdef cppclass cpp_PinnedMemoryResource"rapidsmpf::PinnedMemoryResource":
        void* allocate(cuda_stream_view, size_t) except +ex_handler
        void deallocate(cuda_stream_view, void*, size_t)

cpdef bool_t is_pinned_memory_resources_supported()

cdef class PinnedMemoryResource:
    cdef optional[cpp_PinnedMemoryResource] _handle
