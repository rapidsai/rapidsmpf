# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Memory utilities."""

from __future__ import annotations

# Largest reservation size addressable as a signed 64-bit byte count (INT64_MAX).
_MAX_RESERVATION_BYTES = (1 << 63) - 1


def check_reservation_size(size: int) -> None:
    """
    Reject a reservation size that cannot represent a real allocation.

    A value above ``INT64_MAX`` almost always means the caller computed ``size``
    using an unsigned subtraction that underflowed and wrapped around outside of
    Python. Performing this check in Python, before the value is coerced into a
    ``size_t``, lets us raise a traceback that points at the offending caller
    rather than silently passing the wrapped value down to C++ where it only
    surfaces later as an opaque "value out of range" cast error.

    Parameters
    ----------
    size
        Requested reservation size in bytes.

    Raises
    ------
    ValueError
        If ``size`` exceeds the addressable range.
    """
    if size > _MAX_RESERVATION_BYTES:
        raise ValueError(
            f"reservation size ({size}) exceeds the addressable range; this almost "
            f"certainly indicates an unsigned (size_t) underflow in the caller's size "
            "computation"
        )
