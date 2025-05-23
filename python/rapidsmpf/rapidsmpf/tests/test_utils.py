# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from rapidsmpf.utils.string import format_bytes, parse_boolean, parse_bytes


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
