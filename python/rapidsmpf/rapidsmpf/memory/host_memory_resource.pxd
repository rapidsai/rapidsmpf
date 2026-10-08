# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport int64_t
from libcpp.optional cimport optional
from rmm.librmm.cuda_stream_ref cimport stream_ref

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.memory.scoped_memory_record cimport cpp_ScopedMemoryRecord


cdef extern from "<rapidsmpf/memory/host_memory_resource.hpp>" nogil:
    cdef cppclass cpp_HostMemoryResource"rapidsmpf::HostMemoryResource":
        void* allocate(stream_ref, size_t) except +ex_handler
        void deallocate(stream_ref, void*, size_t)
        int64_t current_allocated() noexcept
        cpp_ScopedMemoryRecord get_main_memory_record() except +ex_handler

cdef class HostMemoryResource:
    cdef optional[cpp_HostMemoryResource] _handle

    @staticmethod
    cdef HostMemoryResource from_handle(
        const optional[cpp_HostMemoryResource]& handle
    )
