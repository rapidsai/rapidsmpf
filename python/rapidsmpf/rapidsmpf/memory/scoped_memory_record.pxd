# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libc.stdint cimport uint64_t

from rapidsmpf._detail.exception_handling cimport ex_handler


cdef extern from "<rapidsmpf/memory/scoped_memory_record.hpp>" nogil:
    cdef cppclass cpp_ScopedMemoryRecord"rapidsmpf::ScopedMemoryRecord":
        cpp_ScopedMemoryRecord() except +ex_handler
        uint64_t num_total_allocs() noexcept
        uint64_t num_current_allocs() noexcept
        uint64_t current() noexcept
        uint64_t total() noexcept
        uint64_t peak() noexcept
        void record_allocation(uint64_t nbytes) noexcept
        void record_deallocation(uint64_t nbytes) noexcept

cdef class ScopedMemoryRecord:
    cdef cpp_ScopedMemoryRecord _handle

    @staticmethod
    cdef ScopedMemoryRecord from_handle(cpp_ScopedMemoryRecord handle)
