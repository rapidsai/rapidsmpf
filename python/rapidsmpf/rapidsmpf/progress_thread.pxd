# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Cython declarations for the ProgressThread interface."""

from libcpp.memory cimport shared_ptr

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.runtime cimport Runtime, cpp_Runtime


cdef extern from "<rapidsmpf/progress_thread.hpp>" nogil:
    cdef cppclass cpp_ProgressThread "rapidsmpf::ProgressThread":
        cpp_ProgressThread(
            shared_ptr[cpp_Runtime] runtime,
        ) except +ex_handler


cdef class ProgressThread:
    cdef shared_ptr[cpp_ProgressThread] _handle
