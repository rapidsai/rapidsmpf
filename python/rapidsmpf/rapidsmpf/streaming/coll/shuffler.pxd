# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, uint32_t
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.span cimport span
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.communicator.communicator cimport Communicator, cpp_Communicator
from rapidsmpf.memory.buffer_resource cimport BufferResource
from rapidsmpf.memory.packed_data cimport cpp_PackedData
from rapidsmpf.shuffler cimport cpp_PartitionOwner
from rapidsmpf.streaming.core.actor cimport cpp_Actor
from rapidsmpf.streaming.core.channel cimport cpp_Channel
from rapidsmpf.streaming.core.context cimport cpp_Context


cdef extern from "<rapidsmpf/streaming/coll/shuffler.hpp>" nogil:
    cdef cpp_Actor cpp_shuffler \
        "rapidsmpf::streaming::actor::shuffler"(
            shared_ptr[cpp_Context] ctx,
            shared_ptr[cpp_Communicator] comm,
            shared_ptr[cpp_Channel] ch_in,
            shared_ptr[cpp_Channel] ch_out,
            int32_t op_id,
            uint32_t total_num_partitions,
            cpp_PartitionOwner partition_owner,
        ) except +ex_handler

    cdef cppclass cpp_ShufflerAsync"rapidsmpf::streaming::ShufflerAsync":
        cpp_ShufflerAsync(
            shared_ptr[cpp_Context] ctx,
            shared_ptr[cpp_Communicator] comm,
            int32_t op_id,
            uint32_t total_num_partitions,
            cpp_PartitionOwner partition_owner,
        ) except +ex_handler
        const shared_ptr[cpp_Communicator]& comm() except +ex_handler
        void insert(unordered_map[uint32_t, cpp_PackedData] chunks) except +ex_handler
        span[const uint32_t] local_partitions() except +ex_handler
        vector[cpp_PackedData] extract(uint32_t pid) except +ex_handler


cdef class ShufflerAsync:
    cdef unique_ptr[cpp_ShufflerAsync] _handle
    cdef Communicator _comm
    cdef BufferResource _br
