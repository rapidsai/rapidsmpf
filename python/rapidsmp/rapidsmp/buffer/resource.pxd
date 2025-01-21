# Copyright (c) 2025, NVIDIA CORPORATION.

from libcpp.memory cimport shared_ptr


cdef extern from "<rapidsmp/buffer/resource.hpp>" nogil:
    cdef cppclass cpp_BufferResource "rapidsmp::BufferResource":
        pass


cdef class BufferResource:
    cdef shared_ptr[cpp_BufferResource] _handle
    cdef cpp_BufferResource* ptr(self)
