# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, uint32_t
from libcpp cimport bool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.communicator.communicator cimport Communicator, cpp_Communicator
from rapidsmpf.memory.buffer_resource cimport (BufferResource,
                                               cpp_BufferResource)
from rapidsmpf.memory.packed_data cimport cpp_PackedData
from rapidsmpf.progress_thread cimport cpp_ProgressThread
from rapidsmpf.statistics cimport cpp_Statistics


cdef extern from "<rapidsmpf/shuffler/shuffler.hpp>" nogil:
    cdef cppclass cpp_Shuffler "rapidsmpf::shuffler::Shuffler":
        cpp_Shuffler(
            shared_ptr[cpp_Communicator] comm,
            shared_ptr[cpp_ProgressThread] comm,
            int32_t op_id,
            uint32_t total_num_partitions,
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
# Insert PackedData into a partition map. We implement this in C++ because
# PackedData doesn't have a default ctor.
cdef extern from *:
    """
    void cpp_insert_chunk_into_partition_map(
        std::unordered_map<std::uint32_t, rapidsmpf::PackedData> &partition_map,
        std::uint32_t pid,
        std::unique_ptr<rapidsmpf::PackedData> packed_data
    ) {
        partition_map.insert({pid, std::move(*packed_data)});
    }
    """
    void cpp_insert_chunk_into_partition_map(
        unordered_map[uint32_t, cpp_PackedData] &partition_map,
        uint32_t pid,
        unique_ptr[cpp_PackedData] packed_data,
    ) except + nogil


cdef class Shuffler:
    cdef unique_ptr[cpp_Shuffler] _handle
    cdef Communicator _comm
    cdef Stream _stream
    cdef BufferResource _br
