# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libc.stdint cimport uint64_t
from libcpp.cast cimport dynamic_cast
from rmm.pylibrmm.memory_resource cimport (DeviceMemoryResource,
                                           UpstreamResourceAdaptor)

# Pointer alias used by `dynamic_cast`, which doesn't accept pointer types.
ctypedef cpp_RmmResourceAdaptor* cpp_RmmResourceAdaptor_ptr


cdef class RmmResourceAdaptor(UpstreamResourceAdaptor):
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
            the primary resource throws `rmm::out_of_memory`.
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

    cdef cpp_RmmResourceAdaptor* get_handle(self):
        cdef cpp_RmmResourceAdaptor* ret = dynamic_cast[cpp_RmmResourceAdaptor_ptr](
            self.c_obj.get()
        )
        assert ret  # The dynamic cast should always succeed.
        return ret

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
