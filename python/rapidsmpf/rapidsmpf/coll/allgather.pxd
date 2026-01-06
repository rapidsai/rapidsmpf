# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int64_t, uint8_t, uint64_t
from libcpp cimport bool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.vector cimport vector
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.communicator.communicator cimport Communicator, cpp_Communicator
from rapidsmpf.memory.buffer_resource cimport (BufferResource,
                                               cpp_BufferResource)
from rapidsmpf.memory.packed_data cimport cpp_PackedData
from rapidsmpf.progress_thread cimport cpp_ProgressThread
from rapidsmpf.statistics cimport cpp_Statistics


cdef extern from "<rapidsmpf/coll/allgather.hpp>" namespace \
        "rapidsmpf::coll::AllGather" nogil:
    cpdef enum class Ordered(bool):
        NO
        YES

cdef extern from "<rapidsmpf/coll/allgather.hpp>" nogil:
    ctypedef int64_t milliseconds_t "std::chrono::milliseconds"

    cdef cppclass cpp_AllGather "rapidsmpf::coll::AllGather":
        cpp_AllGather(
            shared_ptr[cpp_Communicator] comm,
            shared_ptr[cpp_ProgressThread] progress_thread,
            uint8_t op_id,
            cpp_BufferResource *br,
            shared_ptr[cpp_Statistics] statistics
        ) except +
        void insert(uint64_t sequence_number, cpp_PackedData packed_data) except +
        void insert_finished() except +
        bool finished() except +
        vector[cpp_PackedData] wait_and_extract(
            Ordered ordered,
            milliseconds_t timeout
        ) except +
        vector[cpp_PackedData] extract_ready() except +


cdef class AllGather:
    cdef unique_ptr[cpp_AllGather] _handle
    cdef Communicator _comm
    cdef Stream _stream
    cdef BufferResource _br
