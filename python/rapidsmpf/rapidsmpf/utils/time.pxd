# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rapidsmpf._detail.exception_handling cimport ex_handler


cdef extern from "<rapidsmpf/utils/misc.hpp>" nogil:
    cdef cppclass cpp_Duration "rapidsmpf::Duration":
        cpp_Duration() except +ex_handler
        cpp_Duration(double) except +ex_handler
        double count() except +ex_handler
