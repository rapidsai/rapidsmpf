# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int64_t

from rapidsmpf._detail.exception_handling cimport ex_handler


cdef extern from "<rapidsmpf/utils/misc.hpp>" nogil:
    cdef cppclass cpp_Duration "rapidsmpf::Duration":
        cpp_Duration() except +ex_handler
        cpp_Duration(double) except +ex_handler
        double count() except +ex_handler


cdef extern from "<rapidsmpf/utils/time.hpp>" nogil:
    int64_t cpp_event_time_ns "rapidsmpf::event_time_ns"() noexcept
