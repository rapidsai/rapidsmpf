# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint8_t, uint32_t
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from rapidsmpf.memory.packed_data cimport cpp_PackedData
from rapidsmpf.streaming.core.channel cimport cpp_Channel
from rapidsmpf.streaming.core.context cimport cpp_Context
from rapidsmpf.streaming.core.node cimport cpp_Node


cdef extern from "<rapidsmpf/streaming/coll/shuffler.hpp>" nogil:
    cdef cpp_Node cpp_shuffler \
        "rapidsmpf::streaming::node::shuffler"(
            shared_ptr[cpp_Context] ctx,
            shared_ptr[cpp_Channel] ch_in,
            shared_ptr[cpp_Channel] ch_out,
            uint8_t op_id,
            uint32_t total_num_partitions,
        ) except +

    cdef cppclass cpp_ShufflerAsync"rapidsmpf::streaming::ShufflerAsync":
        cpp_ShufflerAsync(
            shared_ptr[cpp_Context] ctx, uint8_t op_id, uint32_t total_num_partitions
        ) except +
        void insert(unordered_map[uint32_t, cpp_PackedData] chunks) except +


cdef class ShufflerAsync:
    cdef unique_ptr[cpp_ShufflerAsync] _handle
