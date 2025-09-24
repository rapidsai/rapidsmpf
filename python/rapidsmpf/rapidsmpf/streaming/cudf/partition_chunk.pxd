# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t, uint64_t
from libcpp.memory cimport unique_ptr
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.buffer.packed_data cimport cpp_PackedData


cdef extern from "<rapidsmpf/streaming/cudf/partition.hpp>" nogil:
    cdef cppclass cpp_PartitionMapChunk "rapidsmpf::streaming::PartitionMapChunk":
        uint64_t sequence_number
        unordered_map[uint32_t, cpp_PackedData] data
        cuda_stream_view stream

    cdef cppclass cpp_PartitionVectorChunk "rapidsmpf::streaming::PartitionVectorChunk":
        uint64_t sequence_number
        vector[cpp_PackedData] data
        cuda_stream_view stream


cdef class PartitionMapChunk:
    cdef unique_ptr[cpp_PartitionMapChunk] _handle
    cdef Stream _stream
    cdef object _owner

    @staticmethod
    cdef PartitionMapChunk from_handle(
        unique_ptr[cpp_PartitionMapChunk] handle, Stream stream, object owner
    )
    cdef const cpp_PartitionMapChunk* handle_ptr(self)
    cdef unique_ptr[cpp_PartitionMapChunk] release_handle(self)


cdef class PartitionVectorChunk:
    cdef unique_ptr[cpp_PartitionVectorChunk] _handle
    cdef Stream _stream
    cdef object _owner

    @staticmethod
    cdef PartitionVectorChunk from_handle(
        unique_ptr[cpp_PartitionVectorChunk] handle, Stream stream, object owner
    )
    cdef const cpp_PartitionVectorChunk* handle_ptr(self)
    cdef unique_ptr[cpp_PartitionVectorChunk] release_handle(self)
