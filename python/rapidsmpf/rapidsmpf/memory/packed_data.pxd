# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0


from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from rapidsmpf.memory.buffer_resource cimport BufferResource


cdef extern from "<rapidsmpf/memory/packed_data.hpp>" nogil:
    cdef cppclass cpp_PackedData "rapidsmpf::PackedData":
        pass


cdef class PackedData:
    cdef unique_ptr[cpp_PackedData] c_obj
    # Prevent the BufferResource (and its stream) from being garbage collected.
    cdef BufferResource _br

    @staticmethod
    cdef from_librapidsmpf(unique_ptr[cpp_PackedData] obj, BufferResource br=*)


cdef list packed_data_vector_to_list(
    vector[cpp_PackedData] packed_data, BufferResource br=*
)
