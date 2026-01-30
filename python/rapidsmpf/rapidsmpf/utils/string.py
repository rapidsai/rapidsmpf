# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Useful string utilities."""

from __future__ import annotations

import re
from typing import TypeVar

T = TypeVar("T")


def format_bytes(nbytes: int | float) -> str:
    """
    Convert a byte size into a human-readable string.

    Parameters
    ----------
    nbytes
        The size in bytes to format.

    Returns
    -------
    str
        A formatted string representing the size with appropriate units.

    Examples
    --------
    >>> format_bytes(100)
    '100 B'
    >>> format_bytes(1024)
    '1 KiB'
    >>> format_bytes(1048576)
    '1 MiB'
    >>> format_bytes(5000000)
    '4.77 MiB'
    >>> format_bytes(1099511627776)
    '1 TiB'
    """
    n = float(nbytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(n) < 1024:
            if n.is_integer():
                return f"{int(n)} {unit}"
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


def parse_bytes(s: str | int) -> int:
    """
    Parse a human-readable byte size string into an integer.

    Parameters
    ----------
    s
        The input size string (e.g., '100 MB', '1KiB', '1e6') or integer.

    Returns
    -------
    int
        The size in bytes.

    Raises
    ------
    TypeError
        Wrong input type.
    ValueError
        If the input cannot be parsed.

    Examples
    --------
    >>> parse_bytes("100")
    100
    >>> parse_bytes("100 MB")
    100000000
    >>> parse_bytes("1KiB")
    1024
    >>> parse_bytes(123)
    123
    >>> parse_bytes("5 foo")
    Traceback (most recent call last):
        ...
    ValueError: Could not parse byte string: '5 foo'
    """
    if isinstance(s, int):
        return s  # Already an integer, return as-is
    if not isinstance(s, str):
        raise TypeError(f"Input must be a string or an integer, got {type(s)}")

    s = s.strip().lower()  # Convert to lowercase for case insensitivity

    # Regex to match size patterns (e.g., "5.4 kB", "100MB", "1e6")
    m = re.fullmatch(r"(\d+(\.\d+)?(?:e[+-]?\d+)?)?\s*([kmgt]?[i]?[b]?)?", s)
    if not m:
        raise ValueError(f"Could not parse byte string: '{s}'")

    number, _, unit = m.groups()
    number = float(number) if number else 1  # Default to 1 if no number provided

    # Unit conversion table
    unit_multipliers = {
        "": 1,
        "b": 1,
        "kb": 10**3,
        "kib": 2**10,
        "mb": 10**6,
        "mib": 2**20,
        "gb": 10**9,
        "gib": 2**30,
        "tb": 10**12,
        "tib": 2**40,
    }

    if unit not in unit_multipliers:
        raise ValueError(f"Unknown unit: '{unit}'")

    return int(number * unit_multipliers[unit])


def parse_bytes_threshold(
    threshold: str | float | None,
    total: int,
    *,
    alignment: int = 1,
) -> int | None:
    """
    Parse a threshold value that can be a fraction, byte count, or byte string.

    Parameters
    ----------
    threshold
        The threshold value. Can be:
        - None: Returns None
        - A float/string in (0, 1]: Interpreted as a fraction of `total`
        - A float/string > 1: Interpreted as an absolute byte count
        - A byte string like "12 GB" or "128 MiB": Parsed as bytes
    total
        The total size (e.g., total device memory) used to compute
        fractional thresholds.
    alignment
        Byte alignment for the result.

    Returns
    -------
    The parsed byte count aligned to `alignment`, or None.

    Examples
    --------
    >>> parse_bytes_threshold(None, 1000)
    >>> parse_bytes_threshold(0.5, 1000)
    500
    >>> parse_bytes_threshold("0.5", 1000)
    500
    >>> parse_bytes_threshold(100, 1000)
    100
    >>> parse_bytes_threshold("12 GB", 1000)
    12000000000
    >>> parse_bytes_threshold(100, 1000, alignment=256)
    0
    """
    if threshold is None:
        return None

    # Try to parse as a byte string (e.g., "12 GB", "128 MiB")
    try:
        if (
            isinstance(threshold, (str, int))
            and (maybe_bytes := parse_bytes(threshold)) >= 1
        ):
            threshold = maybe_bytes
    except (TypeError, ValueError):
        pass

    value = float(threshold)
    if value == 0:
        return None

    if 0.0 < value <= 1.0:
        # Fraction of total
        byte_count = int(total * value)
    else:
        # Absolute byte count
        byte_count = int(value)

    # Align to alignment
    return (byte_count // alignment) * alignment


def parse_boolean(boolean: str) -> bool:
    """
    Parse a string into a boolean value.

    Recognized true values are: "true", "1", "yes", "on"
    Recognized false values are: "false", "0", "no", "off"
    Comparison is case-insensitive and ignores leading/trailing whitespace.

    Parameters
    ----------
    boolean
        The string representation of a boolean value.

    Returns
    -------
    The parsed boolean value.

    Raises
    ------
    ValueError
        If the input string is not a recognized boolean value.
    """
    val = boolean.strip().lower()
    if val in {"true", "1", "yes", "on"}:
        return True
    if val in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from '{boolean}'")
