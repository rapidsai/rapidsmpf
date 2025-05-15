# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.string cimport string


cdef extern from "<rapidsmpf/utils.hpp>" nogil:
    cdef T cpp_parse_string"rapidsmpf::parse_string"[T](string value) except +
