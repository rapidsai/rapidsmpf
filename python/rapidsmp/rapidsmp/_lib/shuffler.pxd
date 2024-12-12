# Copyright (c) 2024, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t
from libcpp.memory cimport shared_ptr, unique_ptr
from rapidsmp._lib.communicator cimport cpp_Communicator

cdef extern from "<rapidsmp/shuffler/shuffler.hpp>" nogil:

    cdef cppclass cpp_Shuffler "rapidsmp::shuffler::Shuffler":
        cpp_Shuffler(
            shared_ptr[cpp_Communicator] comm,
            uint32_t total_num_partitions,
        ) except +
