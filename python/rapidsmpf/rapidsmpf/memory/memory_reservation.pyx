# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libcpp.utility cimport move


cdef class MemoryReservation:
    """
    Represents a reservation for future memory allocation.

    A reservation is created by `BufferResource.reserve` and must be used
    when allocating buffers through the same `BufferResource`.
    """
    def __init__(self):
        raise ValueError("use the `from_handle` factory function")

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @staticmethod
    cdef MemoryReservation from_handle(
        unique_ptr[cpp_MemoryReservation] handle,
        BufferResource br,
    ):
        """
        Construct a MemoryReservation from an existing C++ handle.

        Parameters
        ----------
        handle
            A unique pointer to a C++ MemoryReservation.
        br
            The buffer resource associated with the reservation.

        Returns
        -------
        A new MemoryReservation wrapping the given handle.
        """
        if not handle:
            raise ValueError("handle cannot be None")
        if br is None:
            raise ValueError("br cannot be None")

        cdef MemoryReservation ret = MemoryReservation.__new__(
            MemoryReservation
        )
        ret._handle = move(handle)
        ret._br = br  # Need to keep the buffer resource alive.
        return ret

    @property
    def size(self):
        """
        Get the remaining size of the reserved memory.

        Returns
        -------
        The size of the reserved memory in bytes.
        """
        return deref(self._handle).size()

    @property
    def mem_type(self):
        """
        Get the type of memory associated with this reservation.

        Returns
        -------
        The memory type associated with this reservation.
        """
        return deref(self._handle).mem_type()

    @property
    def br(self):
        """
        Get the buffer resource associated with this reservation.

        Returns
        -------
        The buffer resource associated with this reservation.
        """
        return self._br
