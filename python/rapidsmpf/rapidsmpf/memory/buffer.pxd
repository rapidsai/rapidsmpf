# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libcpp.memory cimport unique_ptr
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.memory.buffer_resource cimport BufferResource


cdef extern from "<rapidsmpf/memory/buffer.hpp>" namespace "rapidsmpf" nogil:
    cpdef enum class MemoryType(int):
        DEVICE
        PINNED_HOST
        HOST

    cdef cppclass cpp_Buffer "rapidsmpf::Buffer":
        size_t size
        # exclusive_data_access() returns std::byte*; void* works because
        # non-const std::byte* -> void* is an implicit C++ conversion.
        void* exclusive_data_access() except +
        void unlock() noexcept
        MemoryType mem_type() noexcept


cdef class BufferHostView:
    cdef Buffer _buf


cdef class Buffer:
    cdef unique_ptr[cpp_Buffer] _handle
    cdef BufferResource _br
    cdef Stream _stream

    @staticmethod
    cdef Buffer from_handle(unique_ptr[cpp_Buffer] handle, BufferResource br, Stream stream)
