# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0


cdef extern from "<rapidsmp/utils.hpp>" nogil:
    cdef cppclass cpp_Duration "rapidsmpf::Duration":
        cpp_Duration() except +
        cpp_Duration(double) except +
