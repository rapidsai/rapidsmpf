# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint8_t, uint32_t
from libcpp cimport bool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.buffer.packed_data cimport cpp_PackedData
from rapidsmpf.buffer.resource cimport BufferResource, cpp_BufferResource
from rapidsmpf.communicator.communicator cimport Communicator, cpp_Communicator
from rapidsmpf.progress_thread cimport cpp_ProgressThread
from rapidsmpf.statistics cimport cpp_Statistics


cdef extern from "<rapidsmpf/shuffler/shuffler.hpp>" nogil:
    cdef cppclass cpp_Shuffler "rapidsmpf::shuffler::Shuffler":
        cpp_Shuffler(
            shared_ptr[cpp_Communicator] comm,
            shared_ptr[cpp_ProgressThread] comm,
            uint8_t op_id,
            uint32_t total_num_partitions,
            cuda_stream_view stream,
            cpp_BufferResource *br,
            shared_ptr[cpp_Statistics] statistics,
        ) except +
        void shutdown() except +
        void insert(unordered_map[uint32_t, cpp_PackedData] chunks) except +
        void concat_insert(unordered_map[uint32_t, cpp_PackedData] chunks) except +
        void insert_finished(vector[uint32_t] pids) except +
        vector[cpp_PackedData] extract(uint32_t pid)  except +
        bool finished() except +
        uint32_t wait_any() except +
        void wait_on(uint32_t pid) except +
        string str() except +

cdef class Shuffler:
    cdef unique_ptr[cpp_Shuffler] _handle
    cdef Communicator _comm
    cdef Stream _stream
    cdef BufferResource _br
