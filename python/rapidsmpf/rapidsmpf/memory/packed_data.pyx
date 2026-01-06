# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libc.stdint cimport uint8_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.contiguous_split cimport PackedColumns
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.device_buffer cimport device_buffer
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.memory.buffer_resource cimport (BufferResource,
                                               cpp_BufferResource)
from rapidsmpf.memory.packed_data cimport cpp_PackedData


# Create a new PackedData from metadata and device buffers.
cdef extern from *:
    """
    std::unique_ptr<rapidsmpf::PackedData> cpp_packed_data_from_buffers(
        std::unique_ptr<std::vector<std::uint8_t>> metadata,
        std::unique_ptr<rmm::device_buffer> gpu_data,
        rmm::cuda_stream_view stream,
        rapidsmpf::BufferResource* br
    ) {
        return std::make_unique<rapidsmpf::PackedData>(
            std::move(metadata), br->move(std::move(gpu_data), stream)
        );
    }
    """
    unique_ptr[cpp_PackedData] cpp_packed_data_from_buffers(
        unique_ptr[vector[uint8_t]] metadata,
        unique_ptr[device_buffer] gpu_data,
        cuda_stream_view stream,
        cpp_BufferResource* br,
    ) except + nogil


cdef class PackedData:
    @staticmethod
    cdef from_librapidsmpf(unique_ptr[cpp_PackedData] obj):
        cdef PackedData self = PackedData.__new__(PackedData)
        self.c_obj = move(obj)
        return self

    @classmethod
    def from_cudf_packed_columns(
        cls,
        PackedColumns packed_columns not None,
        Stream stream not None,
        BufferResource br not None,
    ):
        """
        Constructs a PackedData from CudfPackedColumns by taking the ownership of the
        data and releasing ``packed_columns``.

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
        cdef cuda_stream_view _stream = stream.view()
        cdef cpp_BufferResource* _br = br.ptr()
        cdef PackedData ret = cls.__new__(cls)
        with nogil:
            if not (packed_columns.c_obj != NULL and
                    deref(packed_columns.c_obj).metadata and
                    deref(packed_columns.c_obj).gpu_data):
                raise ValueError("Cannot release empty PackedColumns")

            # we cannot use packed_columns.release() because it returns a tuple of
            # memoryview and gpumemoryview, and we need to take ownership of the
            # underlying buffers
            ret.c_obj = cpp_packed_data_from_buffers(
                move(deref(packed_columns.c_obj).metadata),
                move(deref(packed_columns.c_obj).gpu_data),
                _stream,
                _br,
            )
        return ret

    def __init__(self):
        """Initialize an empty PackedData instance."""
        pass

    def __dealloc__(self):
        with nogil:
            self.c_obj.reset()


# Convert a vector of `cpp_PackedData` into a list of `PackedData`.
cdef list packed_data_vector_to_list(vector[cpp_PackedData] packed_data):
    cdef list ret = []
    for i in range(packed_data.size()):
        ret.append(
            PackedData.from_librapidsmpf(
                make_unique[cpp_PackedData](move(packed_data[i]))
            )
        )
    return ret
