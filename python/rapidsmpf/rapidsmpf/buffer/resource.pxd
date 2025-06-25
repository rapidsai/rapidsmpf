# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libc.stddef cimport size_t
from libc.stdint cimport int64_t
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.optional cimport optional
from libcpp.unordered_map cimport unordered_map
from rmm.librmm.memory_resource cimport device_memory_resource
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
    cdef cppclass cpp_BufferResource "rapidsmpf::BufferResource":
        cpp_BufferResource(
            device_memory_resource *device_mr,
            unordered_map[MemoryType, cpp_MemoryAvailable] memory_available,
            optional[cpp_Duration] periodic_spill_check,
        ) except +
        size_t cpp_memory_reserved "memory_reserved"(
            MemoryType mem_type
        ) except +
        cpp_SpillManager &cpp_spill_manager "spill_manager"() except +


cdef class BufferResource:
    cdef object __weakref__
    cdef shared_ptr[cpp_BufferResource] _handle
    cdef readonly SpillManager spill_manager
    cdef cpp_BufferResource* ptr(self)
    cdef DeviceMemoryResource _mr


cdef extern from "<rapidsmpf/buffer/resource.hpp>" nogil:
    cdef cppclass cpp_LimitAvailableMemory "rapidsmpf::LimitAvailableMemory":
        cpp_LimitAvailableMemory(
            cpp_RmmResourceAdaptor *mr, int64_t limit
        ) except +
        int64_t operator()() except +


cdef class LimitAvailableMemory:
    cdef shared_ptr[cpp_LimitAvailableMemory] _handle
    cdef RmmResourceAdaptor _mr
