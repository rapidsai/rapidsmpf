# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Cython declarations for the ProgressThread interface."""

from cython.operator cimport dereference as deref
from libc.stdint cimport uint64_t, uintptr_t
from libcpp.functional cimport function
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.utility cimport move

from rapidsmpf.communicator.communicator cimport cpp_Logger
from rapidsmpf.statistics cimport cpp_Statistics


cdef extern from "<rapidsmpf/progress_thread.hpp>" nogil:
    cdef cppclass cpp_ProgressThread "rapidsmpf::ProgressThread":
        cpp_ProgressThread(
            cpp_Logger& logger,
            shared_ptr[cpp_Statistics] statistics,
        ) except +


cdef class ProgressThread:
    cdef shared_ptr[cpp_ProgressThread] _handle
