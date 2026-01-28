# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, int64_t, uint64_t
from libcpp cimport bool as bool_t
from libcpp.memory cimport unique_ptr
from libcpp.optional cimport optional
from libcpp.vector cimport vector

from rapidsmpf.streaming.core.message cimport cpp_Message


cdef extern from "<rapidsmpf/streaming/cudf/channel_metadata.hpp>" \
        namespace "rapidsmpf::streaming" nogil:

    cdef cppclass cpp_HashScheme "rapidsmpf::streaming::HashScheme":
        vector[int32_t] column_indices
        int64_t modulus
        cpp_HashScheme() except +
        cpp_HashScheme(vector[int32_t], int64_t) except +
        bool_t operator==(const cpp_HashScheme&)

    cdef enum cpp_SpecType "rapidsmpf::streaming::SpecType":
        NONE "rapidsmpf::streaming::SpecType::NONE"
        ALIGNED "rapidsmpf::streaming::SpecType::ALIGNED"
        HASH "rapidsmpf::streaming::SpecType::HASH"

    cdef cppclass cpp_PartitioningSpec "rapidsmpf::streaming::PartitioningSpec":
        cpp_SpecType type
        optional[cpp_HashScheme] hash

        @staticmethod
        cpp_PartitioningSpec none()

        @staticmethod
        cpp_PartitioningSpec aligned()

        @staticmethod
        cpp_PartitioningSpec from_hash(cpp_HashScheme)

        bool_t operator==(const cpp_PartitioningSpec&)

    cdef cppclass cpp_Partitioning "rapidsmpf::streaming::Partitioning":
        cpp_PartitioningSpec inter_rank
        cpp_PartitioningSpec local
        bool_t operator==(const cpp_Partitioning&)

    cdef cppclass cpp_ChannelMetadata "rapidsmpf::streaming::ChannelMetadata":
        int64_t local_count
        cpp_Partitioning partitioning
        bool_t duplicated
        cpp_ChannelMetadata(
            int64_t,
            cpp_Partitioning,
            bool_t
        ) except +
        bool_t operator==(const cpp_ChannelMetadata&)

    cpp_Message cpp_to_message_channel_metadata "rapidsmpf::streaming::to_message" (
        uint64_t, unique_ptr[cpp_ChannelMetadata]
    ) except +


cdef class HashScheme:
    cdef cpp_HashScheme _scheme

    @staticmethod
    cdef HashScheme from_cpp(cpp_HashScheme scheme)


cdef class Partitioning:
    cdef unique_ptr[cpp_Partitioning] _handle

    @staticmethod
    cdef Partitioning from_handle(unique_ptr[cpp_Partitioning] handle)


cdef class ChannelMetadata:
    cdef unique_ptr[cpp_ChannelMetadata] _handle

    @staticmethod
    cdef ChannelMetadata from_handle(unique_ptr[cpp_ChannelMetadata] handle)

    cdef void _check_handle(self) except *

    cdef unique_ptr[cpp_ChannelMetadata] release_handle(self)
