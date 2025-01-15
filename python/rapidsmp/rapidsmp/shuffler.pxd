# Copyright (c) 2025, NVIDIA CORPORATION.

from cuda.bindings.cyruntime cimport cudaStream_t
from libc.stdint cimport uint32_t
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from pylibcudf.libcudf.contiguous_split cimport packed_columns
from pylibcudf.table cimport Table
from rapidsmp.buffer.resource cimport BufferResource, cpp_BufferResource
from rapidsmp.communicator.communicator cimport Communicator, cpp_Communicator
from rmm._cuda.stream cimport Stream


cpdef dict partition_and_pack(Table table, columns_to_hash, int num_partitions)

cpdef Table unpack_and_concat(partitions)

cdef extern from "<rapidsmp/shuffler/shuffler.hpp>" nogil:
    cdef cppclass cpp_Shuffler "rapidsmp::shuffler::Shuffler":
        cpp_Shuffler(
            shared_ptr[cpp_Communicator] comm,
            uint32_t total_num_partitions,
            cudaStream_t stream,
            cpp_BufferResource *br,
        ) except +
        void insert(unordered_map[uint32_t, packed_columns] chunks) except +
        string str() except +

cdef class Shuffler:
    cdef unique_ptr[cpp_Shuffler] _handle
    cdef Communicator _comm
    cdef Stream _stream
    cdef BufferResource _br
