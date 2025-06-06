# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint8_t
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from rmm.librmm.device_buffer cimport device_buffer


cdef extern from "<rapidsmpf/buffer/packed_data.hpp>" nogil:
    cdef cppclass cpp_PackedData "rapidsmpf::PackedData":
        unique_ptr[vector[uint8_t]] metadata
        unique_ptr[device_buffer] gpu_data

        cpp_PackedData(unique_ptr[vector[uint8_t]] metadata,
                       unique_ptr[device_buffer] gpu_data)


cdef class PackedData:
    cdef unique_ptr[cpp_PackedData] c_obj

    @staticmethod
    cdef from_librapidsmpf(unique_ptr[cpp_PackedData] obj)
