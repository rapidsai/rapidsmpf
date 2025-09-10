# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0


from libc.stdint cimport uint32_t
from libcpp.memory cimport make_unique, shared_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf.types cimport size_type
from rapidsmpf.streaming.core.channel cimport Channel, cpp_Channel
from rapidsmpf.streaming.core.context cimport Context, cpp_Context
from rapidsmpf.streaming.core.node cimport CppNode, cpp_Node


cdef extern from "<rapidsmpf/streaming/cudf/partition.hpp>" nogil:
    int cpp_HASH_MURMUR3"cudf::hash_id::HASH_MURMUR3"
    uint32_t cpp_DEFAULT_HASH_SEED"cudf::DEFAULT_HASH_SEED",
    cdef cpp_Node cpp_partition_and_pack \
        "rapidsmpf::streaming::node::partition_and_pack"(
            shared_ptr[cpp_Context] ctx,
            shared_ptr[cpp_Channel] ch_in,
            shared_ptr[cpp_Channel] ch_out,
            vector[size_type] columns_to_hash,
            int num_partitions,
            int hash_function,
            uint32_t seed,
        ) except +
    cdef cpp_Node cpp_unpack_and_concat \
        "rapidsmpf::streaming::node::unpack_and_concat"(
            shared_ptr[cpp_Context] ctx,
            shared_ptr[cpp_Channel] ch_in,
            shared_ptr[cpp_Channel] ch_out,
        ) except +


def partition_and_pack(
    Context ctx not None,
    Channel ch_in not None,
    Channel ch_out not None,
    object columns_to_hash not None,
    int num_partitions,
):
    cdef vector[size_type] _columns_to_hash = tuple(columns_to_hash)
    cdef cpp_Node _ret
    with nogil:
        _ret = cpp_partition_and_pack(
            ctx._handle,
            ch_in._handle,
            ch_out._handle,
            _columns_to_hash,
            num_partitions,
            cpp_HASH_MURMUR3,
            cpp_DEFAULT_HASH_SEED,
        )
    return CppNode.from_handle(
        make_unique[cpp_Node](move(_ret)), owner = None
    )


def unpack_and_concat(
    Context ctx not None,
    Channel ch_in not None,
    Channel ch_out not None,
):
    cdef cpp_Node _ret
    with nogil:
        _ret = cpp_unpack_and_concat(
            ctx._handle,
            ch_in._handle,
            ch_out._handle,
        )
    return CppNode.from_handle(
        make_unique[cpp_Node](move(_ret)), owner = None
    )
