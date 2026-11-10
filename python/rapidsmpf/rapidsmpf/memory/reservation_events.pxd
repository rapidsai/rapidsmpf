# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint64_t


# Kept in its own module, with no rapidsmpf dependencies, so that it can be cimported
# from `buffer_resource.pxd` and `memory_reservation.pxd`, which cimport each other.
cdef extern from "<rapidsmpf/memory/reservation_events.hpp>" namespace "rapidsmpf":
    ctypedef uint64_t ReservationContext
