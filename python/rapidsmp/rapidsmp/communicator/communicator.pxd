# Copyright (c) 2025, NVIDIA CORPORATION.

from libcpp.memory cimport shared_ptr


cdef extern from "<rapidsmp/communicator/communicator.hpp>" nogil:
    cdef cppclass cpp_Communicator "rapidsmp::Communicator":
        int rank() except +
        int nranks() except +

cdef class Communicator:
    cdef shared_ptr[cpp_Communicator] _handle
