# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource cimport (DeviceMemoryResource,
                                           UpstreamResourceAdaptor,
                                           device_memory_resource)


cdef extern from "<rapidsmpf/rmm_resource_adaptor.hpp>" nogil:
    cdef cppclass cpp_RmmResourceAdaptor"rapidsmpf::RmmResourceAdaptor"(
        device_memory_resource
    ):
        # Notice, `RmmResourceAdaptor` takes `device_async_resource_ref` as
        # upstream_mr arguments but we define them here as `device_memory_resource*`
        # and rely on implicit type conversion.
        cpp_RmmResourceAdaptor(
            device_memory_resource* upstream_mr
        ) except +

        cpp_RmmResourceAdaptor(
            device_memory_resource* upstream_mr,
            device_memory_resource* fallback_mr,
        ) except +


cdef class RmmResourceAdaptor(UpstreamResourceAdaptor):
    cdef readonly DeviceMemoryResource fallback_mr

    def __cinit__(
        self,
        *,
        DeviceMemoryResource upstream_mr,
        DeviceMemoryResource fallback_mr = None,
    ):
        """
        A memory resource that uses an fallback resource when memory allocation fails.
        Parameters
        ----------
        upstream_mr
            The primary resource used for allocating/deallocating device memory.
        fallback_mr
            The fallback resource used when the primary fails to allocate.
        """
        self.fallback_mr = fallback_mr

        if fallback_mr is None:
            self.c_obj.reset(
                new cpp_RmmResourceAdaptor(
                    upstream_mr.get_mr()
                )
            )
        else:
            self.c_obj.reset(
                new cpp_RmmResourceAdaptor(
                    upstream_mr.get_mr(),
                    fallback_mr.get_mr(),
                )
            )

    def __dealloc__(self):
        with nogil:
            self.c_obj.reset()
