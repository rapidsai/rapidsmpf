# # SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# # SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libcpp cimport bool as bool_t

from rapidsmpf.memory.buffer cimport MemoryType as cpp_MemoryType


cdef extern from "<rapidsmpf/memory/content_description.hpp>" nogil:
    cdef cppclass cpp_ContentDescription"rapidsmpf::ContentDescription":
        bool_t spillable() noexcept
        size_t content_size(cpp_MemoryType mem_type) noexcept


cdef content_description_from_cpp(cpp_ContentDescription cd)
