# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""RapidsMPF-specific exception classes."""


# Python exception classes for rapidsmpf-specific exceptions
class ReservationError(MemoryError):
    """
    Exception raised when memory reservation fails.

    This exception is thrown when attempting to reserve memory fails, or when an
    existing reservation is insufficient for a requested allocation. It does not
    necessarily indicate that the system is out of physical memory, only that the
    reservation contract could not be satisfied.
    """
    pass


class OutOfMemory(MemoryError):
    """
    Exception raised when a memory allocation fails due to insufficient memory.

    This exception indicates that the system has run out of the requested
    memory type (device or host) and cannot satisfy the allocation request.
    """
    pass


class BadAlloc(MemoryError):
    """
    Exception raised when a memory allocation fails.

    This is the rapidsmpf equivalent of std::bad_alloc, indicating a failed
    memory allocation operation.
    """
    pass
