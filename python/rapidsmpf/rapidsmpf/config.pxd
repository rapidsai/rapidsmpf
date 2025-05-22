# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map


cdef extern from "<rapidsmpf/config.hpp>" nogil:
    cdef cppclass cpp_Options "rapidsmpf::config::Options":
        cpp_Options() except +
        cpp_Options(unordered_map[string, string] options_as_strings) except +
        unordered_map[string, string] get_strings() except +

cdef class Options:
    cdef cpp_Options _handle
