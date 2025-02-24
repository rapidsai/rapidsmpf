# Copyright (c) 2025, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string


cdef extern from "<rapidsmp/statistics.hpp>" nogil:
    cdef cppclass cpp_Statistics "rapidsmp::Statistics":
        cpp_Statistics() except +
        bool enabled() except +
        string report() except +

cdef class Statistics:
    cdef shared_ptr[cpp_Statistics] _handle
