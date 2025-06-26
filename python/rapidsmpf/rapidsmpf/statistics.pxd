# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

from rapidsmpf.rmm_resource_adaptor cimport RmmResourceAdaptor


cdef extern from "<rapidsmpf/statistics.hpp>" nogil:
    cdef cppclass cpp_Statistics "rapidsmpf::Statistics":
        cpp_Statistics() except +
        bool enabled() except +
        string report() except +
        double add_stat(
            string name,
            double value
        ) except +
        bool is_memory_profiling_enabled() except +

cdef class Statistics:
    cdef shared_ptr[cpp_Statistics] _handle
    cdef RmmResourceAdaptor _mr
