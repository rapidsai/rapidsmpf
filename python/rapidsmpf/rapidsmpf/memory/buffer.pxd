# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libcpp.memory cimport unique_ptr


cdef extern from "<rapidsmpf/memory/buffer.hpp>" namespace "rapidsmpf" nogil:
    cpdef enum class MemoryType(int):
        DEVICE
        PINNED_HOST
        HOST

    cdef cppclass cpp_Buffer "rapidsmpf::Buffer":
        size_t size
        const char* data() except +
        MemoryType mem_type() noexcept


cdef class Buffer:
    cdef unique_ptr[cpp_Buffer] _handle

    @staticmethod
    cdef Buffer from_handle(unique_ptr[cpp_Buffer] handle)
