# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

cdef extern from "<rapidsmpf/streaming/core/channel.hpp>" nogil:
    cdef cppclass cpp_Channel "rapidsmpf::streaming::Channel":
        pass


cdef class Channel:
    cdef cpp_Channel _handle

    @staticmethod
    cdef Channel from_handle(cpp_Channel handle)
