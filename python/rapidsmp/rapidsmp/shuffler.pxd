# Copyright (c) 2025, NVIDIA CORPORATION.

from libc.stdint cimport uint16_t, uint32_t
from libcpp cimport bool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from pylibcudf.libcudf.contiguous_split cimport packed_columns
from pylibcudf.table cimport Table
from rapidsmp.buffer.resource cimport BufferResource, cpp_BufferResource
from rapidsmp.communicator.communicator cimport Communicator, cpp_Communicator
from rapidsmp.statistics cimport cpp_Statistics
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream


cpdef dict partition_and_pack(
    Table table,
    columns_to_hash,
    int num_partitions,
    stream,
    DeviceMemoryResource device_mr,
)

cpdef Table unpack_and_concat(
    partitions,
    stream,
    DeviceMemoryResource device_mr,
)

cdef extern from "<rapidsmp/shuffler/shuffler.hpp>" nogil:
    cdef cppclass cpp_Shuffler "rapidsmp::shuffler::Shuffler":
        cpp_Shuffler(
            shared_ptr[cpp_Communicator] comm,
            uint16_t op_id,
            uint32_t total_num_partitions,
            cuda_stream_view stream,
            cpp_BufferResource *br,
            shared_ptr[cpp_Statistics] statistics,
        ) except +
        void shutdown() except +
        void insert(unordered_map[uint32_t, packed_columns] chunks) except +
        void insert_finished(uint32_t pid) except +
        vector[packed_columns] extract(uint32_t pid)  except +
        bool finished() except +
        uint32_t wait_any() except +
        string str() except +

cdef class Shuffler:
    cdef unique_ptr[cpp_Shuffler] _handle
    cdef Communicator _comm
    cdef Stream _stream
    cdef BufferResource _br
