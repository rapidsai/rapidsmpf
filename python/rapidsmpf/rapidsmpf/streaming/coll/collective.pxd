# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rapidsmpf.streaming.core.context cimport Context


cdef class Collective:
    cdef Context _ctx
    # The finish markers were inserted on this rank.
    cdef bint _finishing
    # aclose() inserted the finish markers because insert_finished() was not called.
    cdef bint _skipped_insert_finished
    cdef bint _closed
    # Shared asyncio.Future that resolves when the collective has completed.
    cdef object _complete

    cdef void _check_open(self) except *
    cdef void _mark_finished(self) except *
    cdef void _cpp_insert_finished(self) except *
    cdef void _cpp_spawn_wait(self, object future) except *
    cdef void _cpp_release(self) noexcept
