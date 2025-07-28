# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libc.stdint cimport uint64_t
from rmm.pylibrmm.memory_resource cimport (DeviceMemoryResource,
                                           UpstreamResourceAdaptor,
                                           device_memory_resource)


cdef extern from "<rapidsmpf/rmm_resource_adaptor.hpp>" nogil:
    cpdef enum class AllocType"rapidsmpf::ScopedMemoryRecord::AllocType"(int):
        PRIMARY
        FALLBACK
        ALL

    cdef cppclass cpp_ScopedMemoryRecord"rapidsmpf::ScopedMemoryRecord":
        cpp_ScopedMemoryRecord() except +
        uint64_t num_total_allocs(AllocType alloc_type) noexcept
        uint64_t num_current_allocs(AllocType alloc_type) noexcept
        uint64_t current(AllocType alloc_type) noexcept
        uint64_t total(AllocType alloc_type) noexcept
        uint64_t peak(AllocType alloc_type) noexcept
        void record_allocation(AllocType alloc_type, uint64_t nbytes) noexcept
        void record_deallocation(AllocType alloc_type, uint64_t nbytes) noexcept

    cdef cppclass cpp_RmmResourceAdaptor"rapidsmpf::RmmResourceAdaptor"(
        device_memory_resource
    ):
        cpp_RmmResourceAdaptor(
            device_memory_resource* upstream_mr
        ) except +

        cpp_RmmResourceAdaptor(
            device_memory_resource* upstream_mr,
            device_memory_resource* fallback_mr,
        ) except +

        cpp_ScopedMemoryRecord get_main_record() except +
        uint64_t current_allocated() noexcept


cdef class RmmResourceAdaptor(UpstreamResourceAdaptor):
    cdef readonly DeviceMemoryResource fallback_mr
    cdef cpp_RmmResourceAdaptor* get_handle(self)


cdef class ScopedMemoryRecord:
    cdef cpp_ScopedMemoryRecord _handle

    @staticmethod
    cdef ScopedMemoryRecord from_handle(cpp_ScopedMemoryRecord handle)
