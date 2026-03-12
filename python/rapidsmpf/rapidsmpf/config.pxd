# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport uint8_t
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from rapidsmpf._detail.exception_handling cimport ex_handler


cdef extern from "<rapidsmpf/config.hpp>" nogil:
    cdef cppclass cpp_Options "rapidsmpf::config::Options":
        cpp_Options() except +ex_handler
        cpp_Options(unordered_map[string, string] options_as_strings) \
            except +ex_handler
        size_t insert_if_absent\
            (unordered_map[string, string] options_as_strings) except +ex_handler
        unordered_map[string, string] get_strings() except +ex_handler
        vector[uint8_t] serialize() except +ex_handler

        @staticmethod
        cpp_Options deserialize(const vector[uint8_t]& buffer) except +ex_handler


cdef class Options:
    cdef cpp_Options _handle
