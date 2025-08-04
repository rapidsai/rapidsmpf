# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0


from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector


cdef extern from "<rapidsmpf/buffer/packed_data.hpp>" nogil:
    cdef cppclass cpp_PackedData "rapidsmpf::PackedData":
        pass


cdef class PackedData:
    cdef unique_ptr[cpp_PackedData] c_obj

    @staticmethod
    cdef from_librapidsmpf(unique_ptr[cpp_PackedData] obj)


cdef list packed_data_vector_to_list(vector[cpp_PackedData] packed_data)
