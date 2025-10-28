# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint64_t
from libcpp cimport bool as bool_t
from libcpp.memory cimport shared_ptr


cdef extern from "<rapidsmpf/streaming/core/channel.hpp>" nogil:
    cdef cppclass cpp_Message"rapidsmpf::streaming::Message":
        void reset() noexcept
        bool_t empty() noexcept
        uint64_t sequence_number() noexcept
        T release[T]() except +
        T& get[T]() except +


cdef class Message:
    cdef cpp_Message _handle

    @staticmethod
    cdef from_handle(cpp_Message handle)
