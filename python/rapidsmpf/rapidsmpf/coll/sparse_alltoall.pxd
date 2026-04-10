# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, int64_t, uint64_t
from libcpp cimport bool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.vector cimport vector
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.communicator.communicator cimport (Communicator, Rank,
                                                  cpp_Communicator)
from rapidsmpf.memory.buffer_resource cimport (BufferResource,
                                               cpp_BufferResource)
from rapidsmpf.memory.packed_data cimport cpp_PackedData
from rapidsmpf.progress_thread cimport cpp_ProgressThread


cdef extern from "<rapidsmpf/coll/sparse_alltoall.hpp>" nogil:
    ctypedef int64_t milliseconds_t "std::chrono::milliseconds"

    cdef cppclass cpp_SparseAlltoall "rapidsmpf::coll::SparseAlltoall":
        cpp_SparseAlltoall(
            shared_ptr[cpp_Communicator] comm,
            int32_t op_id,
            cpp_BufferResource *br,
            vector[Rank] srcs,
            vector[Rank] dsts
        ) except +ex_handler
        void insert(Rank dst, cpp_PackedData packed_data) \
            except +ex_handler
        void insert_finished() except +ex_handler
        const shared_ptr[cpp_Communicator]& comm() except +ex_handler
        void wait(
            milliseconds_t timeout
        ) except +ex_handler
        vector[cpp_PackedData] extract(Rank src) except +ex_handler


cdef class SparseAlltoall:
    cdef unique_ptr[cpp_SparseAlltoall] _handle
    cdef BufferResource _br
    cdef Communicator _comm
