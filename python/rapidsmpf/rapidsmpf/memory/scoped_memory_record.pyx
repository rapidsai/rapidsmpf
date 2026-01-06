# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint64_t


cdef class ScopedMemoryRecord:
    """Scoped memory record for tracking memory usage statistics."""
    @staticmethod
    cdef ScopedMemoryRecord from_handle(cpp_ScopedMemoryRecord handle):
        """Create a new scoped memory record from a C++ handle

        This is copying ``handle``, which is a POD struct.

        Parameters
        ----------
        handle
            C++ handle of a ScopedMemoryRecord.

        Returns
        -------
            Python copy of the scoped memory record.
        """
        cdef ScopedMemoryRecord ret = ScopedMemoryRecord.__new__(ScopedMemoryRecord)
        ret._handle = handle
        return ret

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
