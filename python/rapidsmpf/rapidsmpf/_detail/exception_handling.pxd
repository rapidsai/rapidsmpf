# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.pair cimport pair
from libcpp.string cimport string

ctypedef pair[int, string] CppExcept
cdef CppExcept translate_py_to_cpp_exception(py_exception) noexcept
cdef void throw_py_as_cpp_exception(pair[int, string] res) noexcept nogil
