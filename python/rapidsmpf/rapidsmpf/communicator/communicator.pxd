# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.progress_thread cimport cpp_ProgressThread


cdef extern from "<rapidsmpf/communicator/communicator.hpp>" namespace \
  "rapidsmpf" nogil:
    ctypedef int32_t Rank
    cdef const bint COMM_HAVE_UCXX
    cdef const bint COMM_HAVE_MPI

cdef extern from "<rapidsmpf/communicator/communicator.hpp>" nogil:
    cdef cppclass cpp_Logger "rapidsmpf::Communicator::Logger":
        void log[T](LOG_LEVEL, T msg) except +ex_handler
        LOG_LEVEL verbosity_level() except +ex_handler
    cpdef enum class LOG_LEVEL "rapidsmpf::Communicator::Logger::LOG_LEVEL"(int):
        NONE
        PRINT
        WARN
        INFO
        DEBUG
        TRACE

cdef class Logger:
    cdef shared_ptr[cpp_Logger] _handle

cdef extern from "<rapidsmpf/communicator/communicator.hpp>" nogil:
    cdef cppclass cpp_Communicator "rapidsmpf::Communicator":
        Rank rank() except +ex_handler
        Rank nranks() except +ex_handler
        string str() except +ex_handler
        shared_ptr[cpp_ProgressThread] progress_thread() except +ex_handler
        shared_ptr[cpp_Logger] logger()

cdef class Communicator:
    cdef shared_ptr[cpp_Communicator] _handle
