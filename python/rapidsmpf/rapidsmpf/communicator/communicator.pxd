# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.progress_thread cimport cpp_ProgressThread


cdef class Logger:
    # We hold a weakref to the communicator so that we can report a useful
    # error if attempting to log from a dead communicator
    cdef object _comm
    cdef shared_ptr[cpp_Communicator] handle(self)

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
        Rank rank() except +ex_handler
        Rank nranks() except +ex_handler
        string str() except +ex_handler
        cpp_Logger logger()
        shared_ptr[cpp_ProgressThread] progress_thread()

cdef class Communicator:
    cdef shared_ptr[cpp_Communicator] _handle
    cdef Logger _logger
    cdef object __weakref__
