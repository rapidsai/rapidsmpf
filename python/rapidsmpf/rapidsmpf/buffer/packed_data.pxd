# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0


from libcpp.memory cimport unique_ptr


cdef extern from "<rapidsmpf/buffer/packed_data.hpp>" nogil:
    cdef cppclass cpp_PackedData "rapidsmpf::PackedData":
        cpp_PackedData() except +


cdef class PackedData:
    cdef unique_ptr[cpp_PackedData] c_obj

    @staticmethod
    cdef from_librapidsmpf(unique_ptr[cpp_PackedData] obj)
