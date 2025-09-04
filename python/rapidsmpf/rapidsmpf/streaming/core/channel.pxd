# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool as bool_t
from libcpp.memory cimport shared_ptr


cdef void cython_invoke_python_function(void* py_function) noexcept nogil


cdef extern from "<rapidsmpf/streaming/core/channel.hpp>" nogil:
    cdef cppclass cpp_Message"rapidsmpf::streaming::Message":
        cpp_Message(...) except +
        void reset() noexcept
        bool_t empty() noexcept
        T release[T]() except +
        T& get[T]() except +

    cdef cppclass cpp_Channel"rapidsmpf::streaming::Channel":
        cpp_Channel() except +


cdef class Message:
    cdef cpp_Message _handle

    @staticmethod
    cdef from_handle(cpp_Message handle)


cdef class Channel:
    cdef shared_ptr[cpp_Channel] _handle
