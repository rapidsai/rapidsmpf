# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport int64_t
from libcpp cimport bool as bool_t
from libcpp.memory cimport shared_ptr
from libcpp.optional cimport optional
from libcpp.unordered_map cimport unordered_map
from rmm.librmm.cuda_stream_pool cimport cuda_stream_pool
from rmm.librmm.memory_resource cimport (any_resource, device_accessible,
                                         device_async_resource_ref)
from rmm.pylibrmm.cuda_stream_pool cimport CudaStreamPool
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.config cimport Options, cpp_Options
from rapidsmpf.memory.buffer cimport MemoryType
from rapidsmpf.memory.memory_reservation cimport cpp_MemoryReservation
from rapidsmpf.memory.pinned_memory_resource cimport (PinnedMemoryResource,
                                                      cpp_PinnedMemoryResource)
from rapidsmpf.memory.spill_manager cimport SpillManager, cpp_SpillManager
from rapidsmpf.statistics cimport Statistics, cpp_Statistics
from rapidsmpf.utils.time cimport cpp_Duration


cdef extern from "<rapidsmpf/memory/buffer_resource.hpp>" nogil:
    cdef enum class AllowOverbooking"rapidsmpf::AllowOverbooking"(bool_t):
        NO
        YES

cdef extern from "<rapidsmpf/memory/buffer_resource.hpp>" nogil:
    cdef cppclass cpp_BufferResource "rapidsmpf::BufferResource":
        @staticmethod
        shared_ptr[cpp_BufferResource] create(
            any_resource[device_accessible],
            optional[cpp_PinnedMemoryResource],
            unordered_map[MemoryType, int64_t],
            optional[cpp_Duration],
            shared_ptr[cuda_stream_pool],
            shared_ptr[cpp_Statistics],
        ) except +ex_handler
        size_t memory_reserved(MemoryType mem_type) except +ex_handler
        int64_t memory_available(MemoryType mem_type) except +ex_handler
        void set_memory_limit(MemoryType mem_type, int64_t limit) except +ex_handler
        cpp_SpillManager &spill_manager() except +ex_handler
        const shared_ptr[cuda_stream_pool] &stream_pool() except +ex_handler
        size_t release(cpp_MemoryReservation&, size_t) except +ex_handler
        shared_ptr[cpp_Statistics] statistics() except +ex_handler
        device_async_resource_ref device_mr() noexcept

cdef class BufferResource:
    cdef object __weakref__
    cdef shared_ptr[cpp_BufferResource] _handle
    cdef readonly SpillManager spill_manager
    cdef cpp_BufferResource* ptr(self)
    cdef DeviceMemoryResource _device_mr
    cdef PinnedMemoryResource _pinned_mr
    cdef CudaStreamPool _stream_pool
    cdef Statistics _statistics


cdef class OwningDeviceMemoryResource(DeviceMemoryResource):
    cdef any_resource[device_accessible] c_obj

    @staticmethod
    cdef OwningDeviceMemoryResource _create(
        any_resource[device_accessible] resource,
    )
