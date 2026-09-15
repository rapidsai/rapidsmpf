# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libcpp.utility cimport move

from contextlib import contextmanager

from rapidsmpf._detail.exception_handling cimport ex_handler


cdef extern from * nogil:
    """
    namespace {
    std::unique_ptr<rapidsmpf::MemoryReservation>
    cpp_reservation_split(rapidsmpf::MemoryReservation* res, std::size_t size) {
        return std::make_unique<rapidsmpf::MemoryReservation>(res->split(size));
    }
    }  // namespace
    """
    unique_ptr[cpp_MemoryReservation] cpp_reservation_split(
        cpp_MemoryReservation*,
        size_t,
    ) except +ex_handler


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

    def clear(self):
        """
        Clear the remaining size of the reservation.

        Resets the reservation so that any remaining, unconsumed bytes are released
        back to the underlying memory resource. After this call, the reservation
        has a remaining size of zero and cannot be used to satisfy further
        allocations.
        """
        with nogil:
            deref(self._handle).clear()

    def split(self, size_t size):
        """
        Split off a sub-reservation of ``size`` bytes.

        Reduces this reservation by ``size`` and returns a new reservation of that
        size on the same buffer resource and memory type. The total reserved by the
        buffer resource is unchanged, the bytes are only moved between the two
        reservations.

        Parameters
        ----------
        size
            The number of bytes to split off.

        Returns
        -------
        The new reservation.

        Raises
        ------
        rapidsmpf.error.ReservationError
            If ``size`` exceeds the remaining size of this reservation.

        Examples
        --------
        Scope part of a reservation to the allocation it covers:
        >>> with opaque_memory_usage(reservation.split(scratch_nbytes)):
        ...     # library call that allocates memory unknown to RapidsMPF.
        ...     result = library_op(...)
        >>> # the scratch bytes are released here, the rest of ``reservation``
        >>> # is still available.
        """
        cdef unique_ptr[cpp_MemoryReservation] ret
        with nogil:
            ret = cpp_reservation_split(self._handle.get(), size)
        return MemoryReservation.from_handle(move(ret), self._br)


@contextmanager
def opaque_memory_usage(MemoryReservation reservation not None):
    """
    Associate untracked memory usage with an existing reservation.

    This context manager is intended for code paths that use memory outside of
    RapidsMPF's memory reservation system, for example internal allocations in
    third-party libraries. The memory may be of any type covered by a
    :class:`MemoryReservation`, most commonly device memory.

    While the context is active, the provided memory reservation is considered
    consumed by the enclosed code block. On exit, the reservation is cleared,
    releasing any remaining, unconsumed bytes back to the underlying memory
    resource.

    Parameters
    ----------
    reservation
        Memory reservation that accounts for the untracked memory usage.

    Yields
    ------
    The same reservation, which may be passed to APIs that require an explicit
    reservation object.

    Examples
    --------
    Account for allocations outside RapidsMPF:
    >>> with opaque_memory_usage(reservation):
    ...     # library call that allocates memory unknown to RapidsMPF.
    ...     result = library_op(...)
    """
    try:
        yield reservation
    finally:
        reservation.clear()
