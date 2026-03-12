# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libc.stdint cimport uint64_t
from rmm.librmm.memory_resource cimport device_memory_resource
from rmm.pylibrmm.memory_resource cimport (DeviceMemoryResource,
                                           UpstreamResourceAdaptor)

from rapidsmpf._detail.exception_handling cimport ex_handler


cdef extern from "<rapidsmpf/memory/scoped_memory_record.hpp>" nogil:
    cpdef enum class AllocType"rapidsmpf::ScopedMemoryRecord::AllocType"(int):
        PRIMARY
        FALLBACK
        ALL

    cdef cppclass cpp_ScopedMemoryRecord"rapidsmpf::ScopedMemoryRecord":
        cpp_ScopedMemoryRecord() except +ex_handler
        uint64_t num_total_allocs(AllocType alloc_type) noexcept
        uint64_t num_current_allocs(AllocType alloc_type) noexcept
        uint64_t current(AllocType alloc_type) noexcept
        uint64_t total(AllocType alloc_type) noexcept
        uint64_t peak(AllocType alloc_type) noexcept
        void record_allocation(AllocType alloc_type, uint64_t nbytes) noexcept
        void record_deallocation(AllocType alloc_type, uint64_t nbytes) noexcept

cdef class ScopedMemoryRecord:
    cdef cpp_ScopedMemoryRecord _handle

    @staticmethod
    cdef ScopedMemoryRecord from_handle(cpp_ScopedMemoryRecord handle)
