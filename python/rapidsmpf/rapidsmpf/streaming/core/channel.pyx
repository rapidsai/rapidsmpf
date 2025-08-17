# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.utility cimport move


cdef class Channel:
    @staticmethod
    cdef Channel from_handle(cpp_Channel handle):
        cdef Channel ret = Channel.__new__(Channel)
        ret._handle = move(handle)
        return ret
