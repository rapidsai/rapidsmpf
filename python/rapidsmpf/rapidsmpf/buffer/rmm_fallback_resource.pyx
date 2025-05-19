# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource cimport (DeviceMemoryResource,
                                           UpstreamResourceAdaptor,
                                           device_memory_resource)


cdef extern from "<rapidsmpf/buffer/rmm_fallback_resource.hpp>" nogil:
    cdef cppclass cpp_RmmFallbackResource"rapidsmpf::RmmFallbackResource"(
        device_memory_resource
    ):
        # Notice, `RmmFallbackResource` takes `device_async_resource_ref` as
        # upstream arguments but we define them here as `device_memory_resource*`
        # and rely on implicit type conversion.
        cpp_RmmFallbackResource(
            device_memory_resource* upstream_mr,
            device_memory_resource* alternate_upstream_mr,
        ) except +


cdef class RmmFallbackResource(UpstreamResourceAdaptor):
    cdef readonly DeviceMemoryResource alternate_upstream_mr

    def __cinit__(
        self,
        DeviceMemoryResource upstream_mr,
        DeviceMemoryResource alternate_upstream_mr,
    ):
        if (alternate_upstream_mr is None):
            raise Exception("Argument `alternate_upstream_mr` must not be None")
        self.alternate_upstream_mr = alternate_upstream_mr

        self.c_obj.reset(
            new cpp_RmmFallbackResource(
                upstream_mr.get_mr(),
                alternate_upstream_mr.get_mr(),
            )
        )

    def __init__(
        self,
        DeviceMemoryResource upstream_mr,
        DeviceMemoryResource alternate_upstream_mr,
    ):
        """
        A memory resource that uses an alternate resource when memory allocation fails.
        Parameters
        ----------
        upstream : DeviceMemoryResource
            The primary resource used for allocating/deallocating device memory
        alternate_upstream : DeviceMemoryResource
            The alternate resource used when the primary fails to allocate
        """
        pass

    cpdef DeviceMemoryResource get_alternate_upstream(self):
        return self.alternate_upstream_mr
