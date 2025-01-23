# Copyright (c) 2025, NVIDIA CORPORATION.

from libc.stdint cimport int64_t
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.unordered_map cimport unordered_map
from rapidsmp.buffer.buffer cimport cpp_MemoryType
from rmm.librmm.memory_resource cimport (device_memory_resource,
                                         statistics_resource_adaptor)
from rmm.pylibrmm.memory_resource cimport (DeviceMemoryResource,
                                           StatisticsResourceAdaptor)


cdef extern from "<rapidsmp/buffer/resource.hpp>" nogil:
    ctypedef int64_t (*cpp_MemoryAvailable)()
    cdef cppclass cpp_BufferResource "rapidsmp::BufferResource":
        cpp_BufferResource(
            device_memory_resource *device_mr,
        ) except +
        cpp_BufferResource(
            device_memory_resource *device_mr,
            unordered_map[cpp_MemoryType, cpp_MemoryAvailable] memory_available,
        ) except +


cdef class BufferResource:
    cdef shared_ptr[cpp_BufferResource] _handle
    cdef cpp_BufferResource* ptr(self)


cdef extern from "<rapidsmp/buffer/resource.hpp>" nogil:
    cdef cppclass cpp_LimitAvailableMemory "rapidsmp::LimitAvailableMemory":
        cpp_LimitAvailableMemory(
            statistics_resource_adaptor[device_memory_resource] *mr, int64_t limit
        ) except +
        int64_t operator()() except +


cdef class LimitAvailableMemory:
    cdef unique_ptr[cpp_LimitAvailableMemory] _handle
    cdef StatisticsResourceAdaptor _statistics_mr
