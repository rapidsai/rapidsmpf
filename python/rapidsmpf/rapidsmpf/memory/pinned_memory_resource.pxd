# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool as bool_t
from libcpp.memory cimport shared_ptr


cdef extern from "<rapidsmpf/memory/pinned_memory_resource.hpp>" nogil:
    cdef bool_t cpp_is_pinned_memory_resources_supported \
        "rapidsmpf::is_pinned_memory_resources_supported"(...) except +

    cdef cppclass cpp_PinnedMemoryResource"rapidsmpf::PinnedMemoryResource":
        cpp_PinnedMemoryResource() except +
        cpp_PinnedMemoryResource(int numa_id) except +

cpdef bool_t is_pinned_memory_resources_supported()

cdef class PinnedMemoryResource:
    cdef shared_ptr[cpp_PinnedMemoryResource] _handle
