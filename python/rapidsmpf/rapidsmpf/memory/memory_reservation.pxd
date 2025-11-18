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

from rapidsmpf.memory.buffer cimport MemoryType
from rapidsmpf.memory.buffer_resource cimport BufferResource
from rapidsmpf.memory.spill_manager cimport SpillManager, cpp_SpillManager
from rapidsmpf.rmm_resource_adaptor cimport (RmmResourceAdaptor,
                                             cpp_RmmResourceAdaptor)
from rapidsmpf.statistics cimport Statistics, cpp_Statistics
from rapidsmpf.utils.time cimport cpp_Duration


cdef extern from "<rapidsmpf/memory/memory_reservation.hpp>" nogil:
    cdef cppclass cpp_MemoryReservation "rapidsmpf::MemoryReservation":
        size_t size() noexcept
        MemoryType mem_type() noexcept

cdef class MemoryReservation:
    cdef unique_ptr[cpp_MemoryReservation] _handle
    cdef BufferResource _br

    @staticmethod
    cdef MemoryReservation from_handle(
        unique_ptr[cpp_MemoryReservation] handle,
        BufferResource br,
    )
