# Copyright (c) 2025, NVIDIA CORPORATION.

from cuda.bindings.cyruntime cimport cudaStream_t
from libc.stdint cimport uint32_t
from libcpp.memory cimport shared_ptr, unique_ptr
from rapidsmp.buffer.resource cimport BufferResource, cpp_BufferResource
from rapidsmp.communicator.communicator cimport Communicator, cpp_Communicator
from rmm._cuda.stream cimport Stream


cdef extern from "<rapidsmp/shuffler/shuffler.hpp>" nogil:
    cdef cppclass cpp_Shuffler "rapidsmp::shuffler::Shuffler":
        cpp_Shuffler(
            shared_ptr[cpp_Communicator] comm,
            uint32_t total_num_partitions,
            cudaStream_t stream,
            cpp_BufferResource *br,
        ) except +

cdef class Shuffler:
    cdef unique_ptr[cpp_Shuffler] _handle
    cdef Communicator _comm
    cdef Stream _stream
    cdef BufferResource _br
