# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from rapidsmpf.utils.string import (
    format_bytes,
    parse_boolean,
    parse_bytes,
    parse_bytes_threshold,
)


def test_format_bytes() -> None:
    assert format_bytes(100) == "100 B"
    assert format_bytes(2**10) == "1 KiB"
    assert format_bytes(42.5 * 2**20) == "42.50 MiB"


def test_parse_bytes() -> None:
    assert parse_bytes("100") == 100
    assert parse_bytes("100 MB") == 10**8
    assert parse_bytes("1MiB") == 2**20
    assert parse_bytes("1.42GB") == 1.42 * 10**9
    assert parse_bytes("KB") == 1000

    with pytest.raises(
        ValueError,
        match="Could not parse byte string: '5 foo'",
    ):
        parse_bytes("5 foo")


@pytest.mark.parametrize("input_str", ["true", "TRUE", "  yes ", "1", "On"])
def test_parse_boolean_true(input_str: str) -> None:
    assert parse_boolean(input_str) is True


@pytest.mark.parametrize("input_str", ["false", "FALSE", "  no ", "0", "Off"])
def test_parse_boolean_false(input_str: str) -> None:
    assert parse_boolean(input_str) is False


@pytest.mark.parametrize("invalid_str", ["maybe", "2", "", "none", "yep"])
def test_parse_boolean_invalid(invalid_str: str) -> None:
    with pytest.raises(ValueError, match="Cannot parse boolean"):
        parse_boolean(invalid_str)


# Use a fixed total for deterministic tests
_TOTAL = 16 * 1024 * 1024 * 1024  # 16 GiB


@pytest.mark.parametrize(
    "value,expected",
    [
        # None and zero return None
        (None, None),
        (0, None),
        (0.0, None),
        # Fractions (0 < value <= 1) are % of total
        (0.5, 8 * 1024 * 1024 * 1024),  # 50% of 16 GiB = 8 GiB
        (1.0, _TOTAL),  # 100%
        ("0.5", 8 * 1024 * 1024 * 1024),  # String fraction
        # Values > 1 are byte counts
        (1024 * 1024 * 1024, 1024 * 1024 * 1024),  # 1 GiB
        ("1000000000", 1000000000),  # String byte count
        # Byte strings (e.g., "12 GB", "128 MiB")
        ("1 GiB", 1024 * 1024 * 1024),
        ("12 GB", 12 * 1000 * 1000 * 1000),
    ],
)
def test_parse_bytes_threshold(value: str | float | None, expected: int | None) -> None:
    assert parse_bytes_threshold(value, _TOTAL) == expected


def test_parse_bytes_threshold_alignment() -> None:
    # Results should be aligned to the specified alignment
    assert (
        parse_bytes_threshold(1000, 1000, alignment=256) == 768
    )  # floor(1000 / 256) * 256
    assert (
        parse_bytes_threshold(0.5, 1000, alignment=100) == 500
    )  # 500 is divisible by 100
