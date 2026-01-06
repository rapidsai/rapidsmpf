# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint64_t
from rmm.librmm.memory_resource cimport device_memory_resource
from rmm.pylibrmm.memory_resource cimport (DeviceMemoryResource,
                                           UpstreamResourceAdaptor)

from rapidsmpf.memory.scoped_memory_record cimport cpp_ScopedMemoryRecord


cdef extern from "<rapidsmpf/rmm_resource_adaptor.hpp>" nogil:
    cdef cppclass cpp_RmmResourceAdaptor"rapidsmpf::RmmResourceAdaptor"(
        device_memory_resource
    ):
        cpp_RmmResourceAdaptor(
            device_memory_resource* upstream_mr
        ) except +

        cpp_RmmResourceAdaptor(
            device_memory_resource* upstream_mr,
            device_memory_resource* fallback_mr,
        ) except +

        cpp_ScopedMemoryRecord get_main_record() except +
        uint64_t current_allocated() noexcept


cdef class RmmResourceAdaptor(UpstreamResourceAdaptor):
    cdef readonly DeviceMemoryResource fallback_mr
    cdef cpp_RmmResourceAdaptor* get_handle(self)
