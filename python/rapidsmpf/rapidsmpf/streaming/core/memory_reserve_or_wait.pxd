# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport int64_t
from libcpp.memory cimport shared_ptr

from rapidsmpf.memory.buffer_resource cimport BufferResource


cdef extern from "<rapidsmpf/streaming/core/memory_reserve_or_wait.hpp>" nogil:
    cdef cppclass cpp_MemoryReserveOrWait"rapidsmpf::streaming::MemoryReserveOrWait":
        size_t size() noexcept

    const int64_t cpp_missing_net_memory_delta \
        "rapidsmpf::streaming::MemoryReserveOrWait::missing_net_memory_delta"


cdef class MemoryReserveOrWait:
    cdef shared_ptr[cpp_MemoryReserveOrWait] _handle
    cdef BufferResource _br

    @staticmethod
    cdef MemoryReserveOrWait from_handle(
        shared_ptr[cpp_MemoryReserveOrWait] handle, BufferResource br
    )
