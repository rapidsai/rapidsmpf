# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool as bool_t
from libcpp.memory cimport shared_ptr

from rapidsmpf._detail.exception_handling cimport ex_handler


cdef extern from "<rapidsmpf/memory/pinned_memory_resource.hpp>" nogil:
    cdef cppclass cpp_PinnedMemoryResource"rapidsmpf::PinnedMemoryResource":
        cpp_PinnedMemoryResource() except +ex_handler
        cpp_PinnedMemoryResource(int numa_id) except +ex_handler

cpdef bool_t is_pinned_memory_resources_supported()

cdef class PinnedMemoryResource:
    cdef shared_ptr[cpp_PinnedMemoryResource] _handle
