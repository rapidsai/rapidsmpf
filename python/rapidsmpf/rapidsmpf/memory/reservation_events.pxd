# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint64_t


# Kept in its own module, with no rapidsmpf dependencies, so that it can be cimported
# from `buffer_resource.pxd` and `memory_reservation.pxd`, which cimport each other.
cdef extern from "<rapidsmpf/memory/reservation_events.hpp>" namespace "rapidsmpf":
    cdef cppclass cpp_ReservationContext "rapidsmpf::ReservationContext":
        cpp_ReservationContext() noexcept
        bint is_attributed() noexcept
        uint64_t hash() noexcept

    cdef cppclass cpp_ReservationContextCreator \
            "rapidsmpf::ReservationContextCreator":
        cpp_ReservationContextCreator() noexcept
        cpp_ReservationContext create() noexcept


cdef class ReservationContext:
    cdef cpp_ReservationContext c_obj

    @staticmethod
    cdef ReservationContext from_cpp(cpp_ReservationContext context)


cdef class ReservationContextCreator:
    cdef cpp_ReservationContextCreator c_obj


cdef inline cpp_ReservationContext unwrap_reservation_context(object context) except *:
    if context is None:
        return cpp_ReservationContext()
    if not isinstance(context, ReservationContext):
        raise TypeError(
            "reservation_context must be a ReservationContext or None"
        )
    return (<ReservationContext>context).c_obj
