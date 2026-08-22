# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
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

    def num_total_allocs(self):
        """
        Total number of allocations performed.

        Returns
        -------
        Number of total allocations.
        """
        return self._handle.num_total_allocs()

    def num_current_allocs(self):
        """
        Number of currently active (non-deallocated) allocations.

        Returns
        -------
        Number of active allocations.
        """
        return self._handle.num_current_allocs()

    def current(self):
        """
        Current memory usage in bytes.

        Returns
        -------
        Current memory usage in bytes.
        """
        return self._handle.current()

    def total(self):
        """
        Total number of bytes allocated over the lifetime.

        Returns
        -------
        Total allocated bytes.
        """
        return self._handle.total()

    def peak(self):
        """
        Peak memory usage in bytes.

        Returns
        -------
        Peak memory usage in bytes.
        """
        return self._handle.peak()

    def record_allocation(self, uint64_t nbytes):
        """
        Record a memory allocation event.

        Updates internal statistics for memory allocation and adjusts peak usage
        if the current memory usage exceeds the previous peak.

        Parameters
        ----------
        nbytes
            The number of bytes allocated.
        """
        self._handle.record_allocation(nbytes)

    def record_deallocation(self, uint64_t nbytes):
        """
        Record a memory deallocation event.

        Updates internal statistics to reflect memory being freed.

        Parameters
        ----------
        nbytes
            The number of bytes deallocated.
        """
        self._handle.record_deallocation(nbytes)
