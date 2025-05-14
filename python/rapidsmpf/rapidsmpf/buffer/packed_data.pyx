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

    @classmethod
    def from_cudf_packed_columns(cls, PackedColumns packed_columns):
        """
        Constructs a PackedData from CudfPackedColumns by taking the ownership of the
        data and releasing `packed_columns`.

        Parameters
        ----------
        packed_columns
            Packed data containing metadata and GPU data buffers

        Returns
        -------
        A new PackedData instance containing the packed columns data

        Raises
        ------
        ValueError
            If the PackedColumns object is empty (has been released already).
        """
        cdef PackedData ret = cls.__new__(cls)
        with nogil:
            if not (packed_columns.c_obj != NULL and 
                    deref(packed_columns.c_obj).metadata and
                    deref(packed_columns.c_obj).gpu_data):
                raise ValueError("Cannot release empty PackedColumns")

            # we cannot use packed_columns.release() because it returns a tuple of
            # memoryview and gpumemoryview, and we need to take ownership of the
            # underlying buffers
            ret.c_obj = make_unique[cpp_PackedData](
                move(deref(packed_columns.c_obj).metadata),
                move(deref(packed_columns.c_obj).gpu_data))
        return ret

    def __init__(self):
        """Initialize an empty PackedData instance."""
        pass
