# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport int64_t
from libcpp cimport bool as bool_t
from libcpp.memory cimport shared_ptr
from rmm.librmm.cuda_stream_pool cimport cuda_stream_pool
from rmm.pylibrmm.cuda_stream_pool cimport CudaStreamPool
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from rapidsmpf.memory.buffer cimport MemoryType
from rapidsmpf.memory.memory_reservation cimport cpp_MemoryReservation
from rapidsmpf.memory.pinned_memory_resource cimport PinnedMemoryResource
from rapidsmpf.memory.spill_manager cimport SpillManager, cpp_SpillManager
from rapidsmpf.rmm_resource_adaptor cimport (RmmResourceAdaptor,
                                             cpp_RmmResourceAdaptor)
from rapidsmpf.statistics cimport Statistics, cpp_Statistics
from rapidsmpf.utils.time cimport cpp_Duration


cdef extern from "<rapidsmpf/memory/buffer_resource.hpp>" nogil:
    cdef enum class AllowOverbooking"rapidsmpf::AllowOverbooking"(bool_t):
        NO
        YES

cdef extern from "<functional>" nogil:
    cdef cppclass cpp_MemoryAvailable "std::function<std::int64_t()>":
        pass

cdef extern from "<rapidsmpf/memory/buffer_resource.hpp>" nogil:
    cdef cppclass cpp_BufferResource "rapidsmpf::BufferResource":
        size_t memory_reserved(MemoryType mem_type) except +
        cpp_MemoryAvailable memory_available(MemoryType mem_type) except +
        cpp_SpillManager &spill_manager() except +
        const cuda_stream_pool &stream_pool() except +
        size_t release(cpp_MemoryReservation&, size_t) except +
        shared_ptr[cpp_Statistics] statistics() except +

cdef class BufferResource:
    cdef object __weakref__
    cdef shared_ptr[cpp_BufferResource] _handle
    cdef readonly SpillManager spill_manager
    cdef cpp_BufferResource* ptr(self)
    cdef DeviceMemoryResource _device_mr
    cdef PinnedMemoryResource _pinned_mr
    cdef CudaStreamPool _stream_pool
    cdef Statistics _statistics
    cdef const cuda_stream_pool* stream_pool(self)

cdef extern from "<rapidsmpf/memory/buffer_resource.hpp>" nogil:
    cdef cppclass cpp_LimitAvailableMemory "rapidsmpf::LimitAvailableMemory":
        cpp_LimitAvailableMemory(
            cpp_RmmResourceAdaptor *mr, int64_t limit
        ) except +
        int64_t operator()() except +


cdef class LimitAvailableMemory:
    cdef shared_ptr[cpp_LimitAvailableMemory] _handle
    cdef RmmResourceAdaptor _mr
