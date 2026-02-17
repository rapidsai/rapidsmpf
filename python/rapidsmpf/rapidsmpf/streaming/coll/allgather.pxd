# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, uint64_t
from libcpp.memory cimport shared_ptr, unique_ptr

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.coll.allgather cimport Ordered as cpp_Ordered
from rapidsmpf.memory.packed_data cimport cpp_PackedData
from rapidsmpf.streaming.core.channel cimport cpp_Channel
from rapidsmpf.streaming.core.context cimport cpp_Context
from rapidsmpf.streaming.core.node cimport cpp_Actor


cdef extern from "<rapidsmpf/streaming/coll/allgather.hpp>" nogil:
    cdef cpp_Actor cpp_allgather \
        "rapidsmpf::streaming::actor::allgather"(
            shared_ptr[cpp_Context] ctx,
            shared_ptr[cpp_Channel] ch_in,
            shared_ptr[cpp_Channel] ch_out,
            int32_t op_id,
            cpp_Ordered ordered,
        ) except +ex_handler

    cdef cppclass cpp_AllGather"rapidsmpf::streaming::AllGather":
        cpp_Allgather(
            shared_ptr[cpp_Context] ctx, int32_t op_id
        ) except +ex_handler
        void insert(uint64_t, cpp_PackedData) except +ex_handler
        void insert_finished() except +ex_handler


cdef class AllGather:
    cdef unique_ptr[cpp_AllGather] _handle
