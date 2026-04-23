# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, int64_t, uint64_t
from libcpp cimport bool as bool_t
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.optional cimport optional
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from pylibcudf.libcudf.types cimport null_order as cpp_null_order
from pylibcudf.libcudf.types cimport order as cpp_order
from rmm.librmm.cuda_stream_view cimport cuda_stream_view

from rapidsmpf.streaming.core.message cimport cpp_Message
from rapidsmpf.streaming.cudf.table_chunk cimport TableChunk, cpp_TableChunk


cdef extern from "<rapidsmpf/streaming/cudf/channel_metadata.hpp>" \
        namespace "rapidsmpf::streaming" nogil:

    cdef cppclass cpp_HashScheme "rapidsmpf::streaming::HashScheme":
        vector[int32_t] column_indices
        int modulus
        cpp_HashScheme() except +
        cpp_HashScheme(vector[int32_t], int) except +
        bool_t operator==(const cpp_HashScheme&)

    cdef cppclass cpp_OrderKey "rapidsmpf::streaming::OrderKey":
        int32_t column_index
        cpp_order order
        cpp_null_order null_order
        bool_t operator==(const cpp_OrderKey&)

    cdef cppclass cpp_OrderScheme "rapidsmpf::streaming::OrderScheme":
        vector[cpp_OrderKey] keys
        shared_ptr[cpp_TableChunk] boundaries
        bool_t strict_boundaries
        cpp_OrderScheme() except +
        bool_t operator==(const cpp_OrderScheme&)

    cdef cppclass cpp_PartitioningSpec "rapidsmpf::streaming::PartitioningSpec":
        enum cpp_Type "rapidsmpf::streaming::PartitioningSpec::Type":
            NONE "rapidsmpf::streaming::PartitioningSpec::Type::NONE"
            INHERIT "rapidsmpf::streaming::PartitioningSpec::Type::INHERIT"
            HASH "rapidsmpf::streaming::PartitioningSpec::Type::HASH"
            ORDER "rapidsmpf::streaming::PartitioningSpec::Type::ORDER"

        cpp_Type type
        optional[cpp_HashScheme] hash
        optional[cpp_OrderScheme] order

        @staticmethod
        cpp_PartitioningSpec none()

        @staticmethod
        cpp_PartitioningSpec inherit()

        @staticmethod
        cpp_PartitioningSpec from_hash(cpp_HashScheme)

        @staticmethod
        cpp_PartitioningSpec from_order(cpp_OrderScheme)

        bool_t operator==(const cpp_PartitioningSpec&)

    cdef cppclass cpp_Partitioning "rapidsmpf::streaming::Partitioning":
        cpp_PartitioningSpec inter_rank
        cpp_PartitioningSpec local
        bool_t operator==(const cpp_Partitioning&)

    cdef cppclass cpp_ChannelMetadata "rapidsmpf::streaming::ChannelMetadata":
        uint64_t local_count
        cpp_Partitioning partitioning
        bool_t duplicated
        cpp_ChannelMetadata(
            uint64_t,
            cpp_Partitioning,
            bool_t
        ) except +
        bool_t operator==(const cpp_ChannelMetadata&)

    cpp_Message cpp_to_message_channel_metadata "rapidsmpf::streaming::to_message" (
        uint64_t, unique_ptr[cpp_ChannelMetadata]
    ) except +

    unique_ptr[cpp_ChannelMetadata] channel_metadata_from_message(
        cpp_Message
    ) except +

    cpp_OrderScheme make_order_scheme(
        vector[cpp_OrderKey], unique_ptr[cpp_TableChunk], bool_t
    ) except +


cdef class HashScheme:
    cdef cpp_HashScheme _scheme

    @staticmethod
    cdef HashScheme from_cpp(cpp_HashScheme scheme)


cdef class OrderKey:
    cdef cpp_OrderKey _key

    @staticmethod
    cdef OrderKey from_cpp(cpp_OrderKey key)


cdef class OrderScheme:
    cdef cpp_OrderScheme _storage

    @staticmethod
    cdef OrderScheme from_cpp(cpp_OrderScheme scheme)


cdef class Partitioning:
    # When ``_owner`` is None, ``_ptr == _handle.get()`` (Python-owned partitioning).
    # Otherwise ``_ptr`` aliases storage inside ``_owner`` (e.g. ``ChannelMetadata``).
    cdef unique_ptr[cpp_Partitioning] _handle
    cdef cpp_Partitioning* _ptr
    cdef object _owner

    @staticmethod
    cdef Partitioning from_handle(unique_ptr[cpp_Partitioning] handle)

    @staticmethod
    cdef Partitioning view_of(cpp_Partitioning* ptr, object owner)

    cdef cpp_Partitioning* _get(self)


cdef class ChannelMetadata:
    cdef unique_ptr[cpp_ChannelMetadata] _handle

    @staticmethod
    cdef ChannelMetadata from_handle(unique_ptr[cpp_ChannelMetadata] handle)

    cdef const cpp_ChannelMetadata* handle_ptr(self) except NULL

    cdef unique_ptr[cpp_ChannelMetadata] release_handle(self)
