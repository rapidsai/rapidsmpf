# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Cython declarations for the ProgressThread interface."""

from cython.operator cimport dereference as deref
from libc.stdint cimport uint64_t, uintptr_t
from libcpp.functional cimport function
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.utility cimport move
from rapidsmp.communicator.communicator cimport cpp_Logger
from rapidsmp.statistics cimport cpp_Statistics


cdef extern from "<functional>" nogil:
    cdef cppclass cpp_Function "std::function<rapidsmp::ProgressThread::ProgressState()>":  # noqa: E501
        pass


cdef extern from "<rapidsmp/progress_thread.hpp>" nogil:
    cdef cppclass cpp_ProgressThread "rapidsmp::ProgressThread":
        ctypedef uint64_t FunctionIndex
        ctypedef uintptr_t ProgressThreadAddress
        cppclass FunctionID:
            ProgressThreadAddress thread_address
            FunctionIndex function_index
        enum class ProgressState:
            InProgress
            Done
        ctypedef cpp_Function Function
        cppclass FunctionState:
            Function function
            bint is_done

        cpp_ProgressThread(
            cpp_Logger& logger,
            shared_ptr[cpp_Statistics] statistics,
        ) except +
        void stop()
        FunctionID add_function(Function&& function) except +
        void remove_function(FunctionID function_id) except +


cdef class ProgressThread:
    cdef shared_ptr[cpp_ProgressThread] _handle
