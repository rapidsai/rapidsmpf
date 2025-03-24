# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string


cdef class Logger:
    cdef Communicator _comm

cdef extern from "<rapidsmp/communicator/communicator.hpp>" namespace "rapidsmp" nogil:
    ctypedef int32_t Rank

cdef extern from "<rapidsmp/communicator/communicator.hpp>" nogil:
    cdef cppclass cpp_Communicator "rapidsmp::Communicator":
        Rank rank() except +
        Rank nranks() except +
        string str() except +

cdef extern from "<rapidsmp/communicator/communicator.hpp>" namespace \
  "rapidsmp::Communicator::Logger" nogil:
    cpdef enum class LOG_LEVEL(int):
        NONE
        PRINT
        WARN
        INFO
        DEBUG
        TRACE

cdef class Communicator:
    cdef shared_ptr[cpp_Communicator] _handle
    cdef Logger _logger
