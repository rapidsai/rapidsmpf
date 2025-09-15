# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libcpp cimport bool as bool_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector


cdef extern from "<chrono>" namespace "std::chrono" nogil:
    cdef cppclass milliseconds:
        milliseconds(long long) except +

cdef extern from "<cupti.h>" nogil:
    ctypedef enum CUpti_CallbackId:
        pass


cdef extern from "<rapidsmpf/cupti.hpp>" nogil:
    cdef struct cpp_MemoryDataPoint "rapidsmpf::MemoryDataPoint":
        double timestamp
        size_t free_memory
        size_t total_memory
        size_t used_memory

    cdef cppclass cpp_CuptiMonitor "rapidsmpf::CuptiMonitor":
        cpp_CuptiMonitor(
            bool_t enable_periodic_sampling,
            milliseconds sampling_interval_ms
        ) except +
        void start_monitoring() except +
        void stop_monitoring() except +
        bool_t is_monitoring() except +
        void capture_memory_sample() except +
        const vector[cpp_MemoryDataPoint]& get_memory_samples() except +
        void clear_samples() except +
        size_t get_sample_count() except +
        void write_csv(const string& filename) except +
        void set_debug_output(bool_t enabled, size_t threshold_mb) except +
        unordered_map[CUpti_CallbackId, size_t] get_callback_counters() except +
        void clear_callback_counters() except +
        size_t get_total_callback_count() except +
        string get_callback_summary() except +


cdef class MemoryDataPoint:
    cdef cpp_MemoryDataPoint _data

    @staticmethod
    cdef MemoryDataPoint from_cpp(cpp_MemoryDataPoint data)


cdef class CuptiMonitor:
    cdef unique_ptr[cpp_CuptiMonitor] _handle
