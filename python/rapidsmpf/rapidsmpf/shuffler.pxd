# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, uint32_t
from libcpp cimport bool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.span cimport span
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.communicator.communicator cimport Communicator, cpp_Communicator
from rapidsmpf.memory.buffer_resource cimport (BufferResource,
                                               cpp_BufferResource)
from rapidsmpf.memory.packed_data cimport cpp_PackedData


cdef extern from *:
    """
    enum class PartitionAssignment : std::int32_t {
        ROUND_ROBIN = 0,
        CONTIGUOUS = 1
    };
    """
    cpdef enum class PartitionAssignment(int32_t):
        ROUND_ROBIN
        CONTIGUOUS


cdef extern from "<rapidsmpf/shuffler/shuffler.hpp>" nogil:
    ctypedef int32_t (*cpp_PartitionOwner
                      "rapidsmpf::shuffler::Shuffler::PartitionOwner")(
        const shared_ptr[cpp_Communicator]&, uint32_t, uint32_t
    )
    cdef cppclass cpp_Shuffler "rapidsmpf::shuffler::Shuffler":
        cpp_Shuffler(
            shared_ptr[cpp_Communicator] comm,
            int32_t op_id,
            uint32_t total_num_partitions,
            cpp_BufferResource *br,
            cpp_PartitionOwner partition_owner,
        ) except +ex_handler
        const shared_ptr[cpp_Communicator]& comm() except +ex_handler
        void shutdown() except +ex_handler
        void insert(unordered_map[uint32_t, cpp_PackedData] chunks) \
            except +ex_handler
        void insert_finished() except +ex_handler
        vector[cpp_PackedData] extract(uint32_t pid)  except +ex_handler
        bool finished() except +ex_handler
        void wait() except +ex_handler
        span[const uint32_t] local_partitions() except +ex_handler
        string str() except +ex_handler

        @staticmethod
        int32_t round_robin(const shared_ptr[cpp_Communicator]&, uint32_t, uint32_t)

        @staticmethod
        int32_t contiguous(const shared_ptr[cpp_Communicator]&, uint32_t, uint32_t)


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
    ) except +ex_handler nogil


cdef class Shuffler:
    cdef unique_ptr[cpp_Shuffler] _handle
    cdef BufferResource _br
    cdef Communicator _comm
