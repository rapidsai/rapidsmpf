# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t, uint64_t
from libcpp.memory cimport unique_ptr
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from rapidsmpf.buffer.packed_data cimport cpp_PackedData


cdef extern from "<rapidsmpf/streaming/chunks/partition.hpp>" nogil:
    cdef cppclass cpp_PartitionMapChunk "rapidsmpf::streaming::PartitionMapChunk":
        unordered_map[uint32_t, cpp_PackedData] data

    cdef cppclass cpp_PartitionVectorChunk "rapidsmpf::streaming::PartitionVectorChunk":
        vector[cpp_PackedData] data


cdef class PartitionMapChunk:
    cdef unique_ptr[cpp_PartitionMapChunk] _handle

    @staticmethod
    cdef PartitionMapChunk from_handle(
        unique_ptr[cpp_PartitionMapChunk] handle
    )
    cdef const cpp_PartitionMapChunk* handle_ptr(self)
    cdef unique_ptr[cpp_PartitionMapChunk] release_handle(self)


cdef class PartitionVectorChunk:
    cdef unique_ptr[cpp_PartitionVectorChunk] _handle

    @staticmethod
    cdef PartitionVectorChunk from_handle(
        unique_ptr[cpp_PartitionVectorChunk] handle
    )
    cdef const cpp_PartitionVectorChunk* handle_ptr(self)
    cdef unique_ptr[cpp_PartitionVectorChunk] release_handle(self)
