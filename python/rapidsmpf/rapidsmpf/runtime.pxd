# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport shared_ptr

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.communicator.communicator cimport cpp_Logger
from rapidsmpf.config cimport cpp_Options
from rapidsmpf.statistics cimport cpp_Statistics


cdef extern from "<rapidsmpf/runtime.hpp>" nogil:
    cdef cppclass cpp_Runtime "rapidsmpf::Runtime":
        @staticmethod
        shared_ptr[cpp_Runtime] from_options(cpp_Options options) \
            except +ex_handler

        void reset(cpp_Options new_options) except +ex_handler

        cpp_Options& options() noexcept
        const shared_ptr[cpp_Statistics]& statistics() noexcept
        const shared_ptr[cpp_Logger]& logger() noexcept


cdef class Runtime:
    cdef shared_ptr[cpp_Runtime] _handle
