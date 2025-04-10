# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from rapidsmp.buffer.packed_data cimport cpp_PackedData


cdef class PackedData:
    @staticmethod
    cdef from_librapidsmp(unique_ptr[cpp_PackedData] obj):
        cdef PackedData self = PackedData.__new__(PackedData)
        self.c_obj = move(obj)
        return self
