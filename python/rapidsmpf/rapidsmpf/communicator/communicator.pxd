# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string


cdef class Logger:
    cdef Communicator _comm

cdef extern from "<rapidsmpf/communicator/communicator.hpp>" namespace \
  "rapidsmpf" nogil:
    ctypedef int32_t Rank
    cdef const bint COMM_HAVE_UCXX
    cdef const bint COMM_HAVE_MPI

cdef extern from "<rapidsmpf/communicator/communicator.hpp>" namespace \
  "rapidsmpf::Communicator::Logger" nogil:
    cdef cppclass cpp_Logger:
        pass
    cpdef enum class LOG_LEVEL(int):
        NONE
        PRINT
        WARN
        INFO
        DEBUG
        TRACE

cdef extern from "<rapidsmpf/communicator/communicator.hpp>" nogil:
    cdef cppclass cpp_Communicator "rapidsmpf::Communicator":
        Rank rank() except +
        Rank nranks() except +
        string str() except +
        cpp_Logger logger()

cdef class Communicator:
    cdef shared_ptr[cpp_Communicator] _handle
    cdef Logger _logger
