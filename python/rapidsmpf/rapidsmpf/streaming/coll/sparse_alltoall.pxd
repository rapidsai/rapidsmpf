# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.vector cimport vector

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.communicator.communicator cimport (Communicator, Rank,
                                                  cpp_Communicator)
from rapidsmpf.memory.buffer_resource cimport BufferResource
from rapidsmpf.memory.packed_data cimport cpp_PackedData
from rapidsmpf.streaming.core.context cimport cpp_Context


cdef extern from "<rapidsmpf/streaming/coll/sparse_alltoall.hpp>" nogil:
    cdef cppclass cpp_SparseAlltoall "rapidsmpf::streaming::SparseAlltoall":
        cpp_SparseAlltoall(
            shared_ptr[cpp_Context] ctx,
            shared_ptr[cpp_Communicator] comm,
            int32_t op_id,
            vector[Rank] srcs,
            vector[Rank] dsts,
        ) except +ex_handler
        void insert(Rank dst, cpp_PackedData) except +ex_handler
        const shared_ptr[cpp_Communicator]& comm() except +ex_handler
        const shared_ptr[cpp_Context]& ctx() except +ex_handler
        vector[cpp_PackedData] extract(Rank src) except +ex_handler


cdef class SparseAlltoall:
    cdef unique_ptr[cpp_SparseAlltoall] _handle
    cdef BufferResource _br
    cdef Communicator _comm
