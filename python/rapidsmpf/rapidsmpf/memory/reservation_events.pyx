# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cdef class ReservationContext:
    """An opaque, copyable memory-reservation attribution token."""

    def __init__(self):
        """Create the unattributed context.

        Attributed contexts can only be made by :class:`ReservationContextCreator`.
        """

    @staticmethod
    cdef ReservationContext from_cpp(cpp_ReservationContext context):
        cdef ReservationContext result = ReservationContext.__new__(
            ReservationContext
        )
        result.c_obj = context
        return result

    @property
    def is_attributed(self):
        """Whether this context carries attribution."""
        return self.c_obj.is_attributed()

    def __copy__(self):
        return ReservationContext.from_cpp(self.c_obj)

    def __deepcopy__(self, memo):
        return ReservationContext.from_cpp(self.c_obj)

    def __hash__(self):
        return hash(self.c_obj.hash())

    def __richcmp__(self, other, int op):
        if not isinstance(other, ReservationContext):
            return NotImplemented
        if op == 2:
            return self.c_obj == (<ReservationContext>other).c_obj
        if op == 3:
            return not (self.c_obj == (<ReservationContext>other).c_obj)
        return NotImplemented

    def __repr__(self):
        if not self.c_obj.is_attributed():
            return "ReservationContext(unattributed)"
        return f"ReservationContext(<opaque:{self.c_obj.hash()}>)"


cdef class ReservationContextCreator:
    """Create globally unique reservation attribution contexts."""

    def create(self):
        """Create a new context."""
        return ReservationContext.from_cpp(self.c_obj.create())
