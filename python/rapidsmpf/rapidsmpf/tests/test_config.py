# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from rapidsmpf.config import Options


def test_get_or_default_with_explicit_values() -> None:
    opts = Options(
        {"debug": "true", "max_retries": "3", "timeout": "2.5", "mode": "fast"}
    )

    assert opts.get_or_assign("debug", bool, default_value=False) is True
    assert opts.get_or_assign("max_retries", int, default_value=0) == 3
    assert opts.get_or_assign("timeout", float, default_value=0.0) == 2.5
    assert opts.get_or_assign("mode", str, default_value="slow") == "fast"


def test_get_or_assign_returns_default_when_key_missing() -> None:
    opts = Options({})

    assert opts.get_or_assign("use_gpu", bool, default_value=True) is True
    assert opts.get_or_assign("workers", int, default_value=4) == 4
    assert opts.get_or_assign("rate", float, default_value=1.2) == 1.2
    assert opts.get_or_assign("name", str, default_value="default") == "default"


def test_get_or_assign_caches_assigned_value() -> None:
    opts = Options({})

    val1 = opts.get_or_assign("threshold", float, default_value=0.75)
    val2 = opts.get_or_assign("threshold", float, default_value=1.23)

    assert val1 == val2 == 0.75  # Second call should not override the first


def test_get_or_assign_raises_on_unsupported_type() -> None:
    class Unsupported:
        pass

    opts = Options({})
    with pytest.raises(ValueError, match="is not support"):
        opts.get_or_assign("key", Unsupported, Unsupported())


def test_get_or_assign_type_conflict_on_same_key() -> None:
    opts = Options({"batch_size": "32"})

    # First access with int parser
    val = opts.get_or_assign("batch_size", int, default_value=16)
    assert val == 32

    # Now try to access same key with a different type
    with pytest.raises(ValueError, match="incompatible template type"):
        opts.get_or_assign("batch_size", float, default_value=32.0)
