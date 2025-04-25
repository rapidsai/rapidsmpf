# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move
from pylibcudf.contiguous_split cimport PackedColumns

from rapidsmpf.buffer.packed_data cimport cpp_PackedData


cdef class PackedData:
    @staticmethod
    cdef from_librapidsmpf(unique_ptr[cpp_PackedData] obj):
        cdef PackedData self = PackedData.__new__(PackedData)
        self.c_obj = move(obj)
        return self

    def __init__(self, PackedColumns packed_columns) -> None:
        """
        Constructs a PackedData from cudf PackedColumns by taking the ownership of the
        data.

        Parameters
        ----------
        packed_columns
            Packed data what contains metadata and GPU data buffers
        """
        with nogil:
            self.c_obj = make_unique[cpp_PackedData](
                move(deref(packed_columns.c_obj).metadata),
                move(deref(packed_columns.c_obj).gpu_data))
