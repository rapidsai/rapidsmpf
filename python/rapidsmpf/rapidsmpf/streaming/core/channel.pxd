# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport shared_ptr

from rapidsmpf._detail.exception_handling cimport ex_handler


cdef extern from "<rapidsmpf/streaming/core/channel.hpp>" nogil:
    cdef cppclass cpp_Channel"rapidsmpf::streaming::Channel":
        cpp_Channel() except +ex_handler

cdef class Channel:
    cdef shared_ptr[cpp_Channel] _handle
    cdef object _on_send
    cdef object _on_recv

    @staticmethod
    cdef from_handle(shared_ptr[cpp_Channel] ch)
