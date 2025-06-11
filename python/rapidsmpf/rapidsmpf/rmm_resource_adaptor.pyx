# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libc.stdint cimport uint64_t
from libcpp.cast cimport dynamic_cast
from rmm.pylibrmm.memory_resource cimport (DeviceMemoryResource,
                                           UpstreamResourceAdaptor)

# Pointer alias used by `dynamic_cast`, which doesn't accept pointer types.
ctypedef cpp_RmmResourceAdaptor* cpp_RmmResourceAdaptor_ptr

cdef class ScopedMemoryRecord:
    cdef cpp_ScopedMemoryRecord _handle

    def __cinit__(self):
        self._handle = cpp_ScopedMemoryRecord()

    def num_total_allocs(self, AllocType alloc_type=AllocType.ALL):
        """
        Total number of allocations performed.

        Parameters
        ----------
        alloc_type
            Allocator type to query. Defaults to ALL.

        Returns
        -------
        Number of total allocations.
        """
        return self._handle.num_total_allocs(alloc_type)

    def num_current_allocs(self, AllocType alloc_type=AllocType.ALL):
        """
        Number of currently active (non-deallocated) allocations.

        Parameters
        ----------
        alloc_type
            Allocator type to query. Defaults to ALL.

        Returns
        -------
        Number of active allocations.
        """
        return self._handle.num_current_allocs(alloc_type)

    def current(self, AllocType alloc_type=AllocType.ALL):
        """
        Current memory usage in bytes.

        Parameters
        ----------
        alloc_type
            Allocator type to query. Defaults to ALL.

        Returns
        -------
        Current memory usage in bytes.
        """
        return self._handle.current(alloc_type)

    def total(self, AllocType alloc_type=AllocType.ALL):
        """
        Total number of bytes allocated over the lifetime.

        Parameters
        ----------
        alloc_type
            Allocator type to query. Defaults to ALL.

        Returns
        -------
        Total allocated bytes.
        """
        return self._handle.total(alloc_type)

    def peak(self, AllocType alloc_type=AllocType.ALL):
        """
        Peak memory usage in bytes.

        Parameters
        ----------
        alloc_type
            Allocator type to query. Defaults to ALL.

        Returns
        -------
        Peak memory usage in bytes.
        """
        return self._handle.peak(alloc_type)

    def record_allocation(self, AllocType alloc_type, uint64_t nbytes):
        """
        Record a memory allocation event.

        Updates internal statistics for memory allocation and adjusts peak usage
        if the current memory usage exceeds the previous peak.

        Parameters
        ----------
        alloc_type
            The allocator that performed the allocation.

        nbytes
            The number of bytes allocated.
        """
        self._handle.record_allocation(alloc_type, nbytes)

    def record_deallocation(self, AllocType alloc_type, uint64_t nbytes):
        """
        Record a memory deallocation event.

        Updates internal statistics to reflect memory being freed.

        Parameters
        ----------
        alloc_type
            The allocator that performed the deallocation.

        nbytes
            The number of bytes deallocated.
        """
        self._handle.record_deallocation(alloc_type, nbytes)


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

    def get_main_record(self):
        """Get a copy of the tracked main record.

        Returns
        -------
        Scoped memory record instance.
        """
        cdef cpp_RmmResourceAdaptor* mr = self.get_handle()
        cdef ScopedMemoryRecord ret = ScopedMemoryRecord.__new__(ScopedMemoryRecord)
        with nogil:
            ret._handle = deref(mr).get_main_record()
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
