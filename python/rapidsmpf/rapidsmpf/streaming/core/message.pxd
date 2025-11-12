# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from libcpp cimport bool as bool_t
from libcpp.memory cimport shared_ptr

from rapidsmpf.buffer.content_description cimport cpp_ContentDescription
from rapidsmpf.buffer.resource cimport cpp_MemoryReservation


cdef extern from "<rapidsmpf/streaming/core/channel.hpp>" nogil:
    cdef cppclass cpp_Message"rapidsmpf::streaming::Message":
        void reset() noexcept
        bool_t empty() noexcept
        uint64_t sequence_number() noexcept
        cpp_ContentDescription content_description() noexcept
        size_t copy_cost() noexcept
        cpp_Message copy(cpp_MemoryReservation& reservation) except +
        T release[T]() except +
        T& get[T]() except +


cdef class Message:
    cdef cpp_Message _handle

    @staticmethod
    cdef from_handle(cpp_Message handle)
