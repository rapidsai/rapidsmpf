# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport int64_t, uint64_t
from libcpp cimport bool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.memory.scoped_memory_record cimport cpp_ScopedMemoryRecord
from rapidsmpf.rmm_resource_adaptor cimport (RmmResourceAdaptor,
                                             cpp_RmmResourceAdaptor)


cdef extern from "<rapidsmpf/statistics.hpp>" nogil:
    cdef cppclass cpp_Statistics "rapidsmpf::Statistics":
        bool enabled() except +ex_handler
        string report() except +ex_handler
        double add_stat(
            string name,
            double value
        ) except +ex_handler
        bool is_memory_profiling_enabled() except +ex_handler
        unordered_map[string, cpp_MemoryRecord] get_memory_records() \
            except +ex_handler

    cdef struct cpp_MemoryRecord "rapidsmpf::Statistics::MemoryRecord":
        cpp_ScopedMemoryRecord scoped
        int64_t global_peak
        uint64_t num_calls

    cdef cppclass cpp_MemoryRecorder "rapidsmpf::Statistics::MemoryRecorder":
        cpp_MemoryRecorder(
            cpp_Statistics* stats,
            cpp_RmmResourceAdaptor* mr,
            string name
        ) except +ex_handler


cdef class Statistics:
    cdef shared_ptr[cpp_Statistics] _handle
    cdef RmmResourceAdaptor _mr


cdef class MemoryRecorder:
    cdef unique_ptr[cpp_MemoryRecorder] _handle
    cdef Statistics _stats
    cdef RmmResourceAdaptor _mr
    cdef string _name
