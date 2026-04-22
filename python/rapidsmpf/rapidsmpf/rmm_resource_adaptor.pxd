# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint64_t
from libcpp.memory cimport unique_ptr
from libcpp.optional cimport optional
from rmm.librmm.memory_resource cimport (any_resource, device_accessible,
                                         device_async_resource_ref,
                                         make_device_async_resource_ref)
from rmm.pylibrmm.memory_resource cimport (DeviceMemoryResource,
                                           UpstreamResourceAdaptor)

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.memory.scoped_memory_record cimport cpp_ScopedMemoryRecord


cdef extern from "<rapidsmpf/rmm_resource_adaptor.hpp>" nogil:
    cdef cppclass cpp_RmmResourceAdaptor"rapidsmpf::RmmResourceAdaptor":
        cpp_RmmResourceAdaptor(
            any_resource[device_accessible] primary_mr,
        ) except +ex_handler

        cpp_RmmResourceAdaptor(
            any_resource[device_accessible] primary_mr,
            optional[any_resource[device_accessible]] fallback_mr,
        ) except +ex_handler

        cpp_ScopedMemoryRecord get_main_record() except +ex_handler
        uint64_t current_allocated() noexcept


# The make_device_async_resource_ref C++ template (declared in RMM's
# memory_resource.pxd) also covers RmmResourceAdaptor; declare the overload.
cdef extern from *:
    optional[device_async_resource_ref] make_device_async_resource_ref(
        cpp_RmmResourceAdaptor&) except +


cdef class RmmResourceAdaptor(UpstreamResourceAdaptor):
    cdef unique_ptr[cpp_RmmResourceAdaptor] c_obj
    cdef readonly DeviceMemoryResource fallback_mr
    cdef cpp_RmmResourceAdaptor* get_handle(self)
