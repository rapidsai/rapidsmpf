# Copyright (c) 2025, NVIDIA CORPORATION.
from __future__ import annotations

import pytest

from rapidsmp.utils.string import format_bytes, parse_bytes


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
