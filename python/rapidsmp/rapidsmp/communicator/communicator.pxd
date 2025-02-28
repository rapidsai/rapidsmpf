# Copyright (c) 2025, NVIDIA CORPORATION.

from libcpp.memory cimport shared_ptr
from libcpp.string cimport string


cdef class Logger:
    cdef Communicator _comm

cdef extern from "<rapidsmp/communicator/communicator.hpp>" nogil:
    cdef cppclass cpp_Communicator "rapidsmp::Communicator":
        int rank() except +
        int nranks() except +
        string str() except +

cdef class Communicator:
    cdef shared_ptr[cpp_Communicator] _handle
    cdef Logger _logger
