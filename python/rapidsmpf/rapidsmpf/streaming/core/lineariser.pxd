# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

from rapidsmpf.streaming.core.channel cimport cpp_Channel


cdef extern from "<rapidsmpf/streaming/core/lineariser.hpp>" nogil:
    cdef cppclass cpp_Lineariser"rapidsmpf::streaming::Lineariser":
        cpp_Lineariser(shared_ptr[cpp_Channel], size_t) except +
        vector[shared_ptr[cpp_Channel]]& get_inputs() except +

cdef class Lineariser:
    cdef shared_ptr[cpp_Lineariser] _handle
