# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libc.stddef cimport size_t
from libc.stdint cimport int64_t
from libcpp cimport bool as bool_t
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.optional cimport optional
from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map
from rmm.librmm.cuda_stream_pool cimport cuda_stream_pool
from rmm.librmm.memory_resource cimport device_memory_resource
from rmm.pylibrmm.cuda_stream_pool cimport CudaStreamPool
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from rapidsmpf.buffer.buffer cimport MemoryType
from rapidsmpf.buffer.spill_manager cimport SpillManager, cpp_SpillManager
from rapidsmpf.rmm_resource_adaptor cimport (RmmResourceAdaptor,
                                             cpp_RmmResourceAdaptor)
from rapidsmpf.utils.time cimport cpp_Duration


cdef extern from "<functional>" nogil:
    cdef cppclass cpp_MemoryAvailable "std::function<std::int64_t()>":
        pass

cdef extern from "<rapidsmpf/buffer/resource.hpp>" nogil:
    cdef cppclass cpp_MemoryReservation "rapidsmpf::MemoryReservation":
        size_t size() noexcept
        MemoryType mem_type() noexcept

    cdef cppclass cpp_BufferResource "rapidsmpf::BufferResource":
        cpp_BufferResource(
            device_memory_resource *device_mr,
            unordered_map[MemoryType, cpp_MemoryAvailable] memory_available,
            optional[cpp_Duration] periodic_spill_check,
            shared_ptr[cuda_stream_pool] stream_pool,
        ) except +
        size_t memory_reserved(MemoryType mem_type) except +
        cpp_MemoryAvailable memory_available(MemoryType mem_type) except +
        cpp_SpillManager &spill_manager() except +
        const cuda_stream_pool &stream_pool() except +
        size_t release(cpp_MemoryReservation&, size_t) except +


cdef class MemoryReservation:
    cdef unique_ptr[cpp_MemoryReservation] _handle
    cdef BufferResource _br

    @staticmethod
    cdef MemoryReservation from_handle(
        unique_ptr[cpp_MemoryReservation] handle,
        BufferResource br,
    )

cdef class BufferResource:
    cdef object __weakref__
    cdef shared_ptr[cpp_BufferResource] _handle
    cdef readonly SpillManager spill_manager
    cdef cpp_BufferResource* ptr(self)
    cdef DeviceMemoryResource _mr
    cdef CudaStreamPool _stream_pool
    cdef const cuda_stream_pool* stream_pool(self)

cdef extern from "<rapidsmpf/buffer/resource.hpp>" nogil:
    cdef cppclass cpp_LimitAvailableMemory "rapidsmpf::LimitAvailableMemory":
        cpp_LimitAvailableMemory(
            cpp_RmmResourceAdaptor *mr, int64_t limit
        ) except +
        int64_t operator()() except +


cdef class LimitAvailableMemory:
    cdef shared_ptr[cpp_LimitAvailableMemory] _handle
    cdef RmmResourceAdaptor _mr
