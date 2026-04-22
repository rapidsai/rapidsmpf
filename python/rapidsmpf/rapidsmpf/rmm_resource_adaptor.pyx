# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libc.stdint cimport uint64_t
from libcpp.optional cimport optional
from rmm.librmm.memory_resource cimport (any_resource, device_accessible,
                                         make_any_device_resource)
from rmm.pylibrmm.memory_resource cimport (DeviceMemoryResource,
                                           UpstreamResourceAdaptor)

from rapidsmpf.memory.scoped_memory_record cimport ScopedMemoryRecord


cdef class RmmResourceAdaptor(UpstreamResourceAdaptor):
    """A RMM memory resource adaptor tailored to RapidsMPF."""
    def __cinit__(
        self,
        DeviceMemoryResource upstream_mr,
        *,
        DeviceMemoryResource fallback_mr = None,
    ):
        """
        A RMM memory resource adaptor tailored to RapidsMPF.

        This adaptor implements:
        - Memory usage tracking.
        - Fallback memory resource support upon out-of-memory in the primary resource.

        Parameters
        ----------
        upstream_mr
            The primary device memory resource used for allocations and deallocations.

        fallback_mr
            If specified, a fallback device memory resource used when allocation from
            the primary resource throws ``rmm::out_of_memory``.
        """
        self.fallback_mr = fallback_mr

        if fallback_mr is None:
            self.c_obj.reset(
                new cpp_RmmResourceAdaptor(
                    make_any_device_resource(upstream_mr.get_mr())
                )
            )
        else:
            self.c_obj.reset(
                new cpp_RmmResourceAdaptor(
                    make_any_device_resource(upstream_mr.get_mr()),
                    optional[any_resource[device_accessible]](
                        make_any_device_resource(fallback_mr.get_mr())
                    ),
                )
            )
        self.c_ref = make_device_async_resource_ref(deref(self.c_obj))

    def __dealloc__(self):
        with nogil:
            self.c_obj.reset()

    cdef cpp_RmmResourceAdaptor* get_handle(self):
        return self.c_obj.get()

    def get_main_record(self):
        """Returns a copy of the main memory record.

        The main record tracks memory statistics for the lifetime of the resource.

        Returns
        -------
        A copy of the current main memory record.
        """
        cdef cpp_RmmResourceAdaptor* mr = self.get_handle()
        cdef cpp_ScopedMemoryRecord ret
        with nogil:
            ret = deref(mr).get_main_record()
        return ScopedMemoryRecord.from_handle(ret)

    @property
    def current_allocated(self) -> int:
        """Get the total number of currently allocated bytes.

        This includes both allocations on the primary and fallback memory resources.

        Returns
        -------
        Total number of currently allocated bytes.
        """
        cdef cpp_RmmResourceAdaptor* mr = self.get_handle()
        cdef uint64_t ret
        with nogil:
            ret = deref(mr).current_allocated()
        return ret
