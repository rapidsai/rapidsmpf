# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from libcpp.memory cimport unique_ptr

from rapidsmpf.memory.buffer cimport MemoryType
from rapidsmpf.memory.buffer_resource cimport BufferResource


cdef extern from "<rapidsmpf/memory/memory_reservation.hpp>" nogil:
    cdef cppclass cpp_MemoryReservation "rapidsmpf::MemoryReservation":
        void clear() noexcept
        size_t size() noexcept
        MemoryType mem_type() noexcept
        uint64_t reservation_context() noexcept

cdef class MemoryReservation:
    cdef unique_ptr[cpp_MemoryReservation] _handle
    cdef BufferResource _br

    @staticmethod
    cdef MemoryReservation from_handle(
        unique_ptr[cpp_MemoryReservation] handle,
        BufferResource br,
    )
