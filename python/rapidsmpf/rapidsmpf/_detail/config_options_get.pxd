# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool as bool_t

from rapidsmpf.config cimport Options


cdef bool_t get_bool(Options options, str key, factory)
cdef int get_int(Options options, str key, factory)
cdef float get_float(Options options, str key, factory)
cdef str get_str(Options options, str key, factory)
