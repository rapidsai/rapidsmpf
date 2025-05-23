# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pickle

import pytest

from rapidsmpf.config import Options


def test_get_or_default_with_explicit_values() -> None:
    opts = Options(
        {
            "debug": "true",
            "max_retries": "3",
            "timeout": "2.5",
            "int": "2.5",
            "mode": "fast",
        }
    )
    assert opts.get_or_assign("debug", bool, default_value=False) is True
    assert opts.get_or_assign("max_retries", int, default_value=0) == 3
    assert opts.get_or_assign("int", int, default_value=0) == 2
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


def test_get_or_default_int64_overflow() -> None:
    opts = Options({"large_int": str(2**65)})
    with pytest.raises(ValueError, match='cannot parse "36893488147419103232"'):
        opts.get_or_assign("large_int", int, default_value=0)

    with pytest.raises(
        OverflowError, match="Python int too large to convert to C long"
    ):
        opts.get_or_assign("another_large_int", int, default_value=2**65)


def test_get_retrieves_existing_values() -> None:
    opts = Options(
        {
            "debug": "true",
            "max_retries": "3",
            "timeout": "2.5",
            "mode": "fast",
        }
    )
    assert opts.get("debug", bool, lambda s: s == "true") is True
    assert opts.get("max_retries", int, int) == 3
    assert opts.get("timeout", float, float) == 2.5
    assert opts.get("mode", str, str) == "fast"


def test_get_uses_factory_when_missing() -> None:
    opts = Options({})
    assert opts.get("feature_enabled", bool, lambda s: True) is True
    assert opts.get("retries", int, lambda s: 5) == 5
    assert opts.get("threshold", float, lambda s: 0.85) == 0.85
    assert opts.get("profile", str, lambda s: "standard") == "standard"


def test_get_caches_value_after_first_use() -> None:
    opts = Options({})

    def factory_1(_: str) -> int:
        return 42

    def factory_2(_: str) -> int:
        return 999

    val1 = opts.get("id", int, factory_1)
    val2 = opts.get("id", int, factory_2)
    assert val1 == val2 == 42


def test_get_raises_on_type_conflict() -> None:
    opts = Options({"batch_size": "64"})

    assert opts.get("batch_size", int, int) == 64
    with pytest.raises(ValueError, match="incompatible template type"):
        opts.get("batch_size", float, float)


def test_get_raises_on_unsupported_type() -> None:
    class Unsupported:
        pass

    opts = Options({})
    with pytest.raises(ValueError, match="is not supported"):
        opts.get("weird", Unsupported, lambda s: Unsupported())


def test_get_strings_returns_correct_data() -> None:
    input_data = {"Alpha": "one", "BETA": "2", "gamma": "THREE"}

    opts = Options(input_data)
    result = opts.get_strings()

    # Keys should be lowercase
    expected_keys = {k.lower() for k in input_data}
    assert set(result.keys()) == expected_keys

    for k, v in input_data.items():
        assert result[k.lower()] == v


def test_get_strings_returns_empty_dict_for_empty_options() -> None:
    opts = Options()
    result = opts.get_strings()
    assert result == {}


def test_get_strings_is_idempotent() -> None:
    opts = Options({"key": "value"})
    result1 = opts.get_strings()
    result2 = opts.get_strings()

    assert result1 == result2
    assert result1["key"] == "value"


def test_serialize_deserialize_roundtrip() -> None:
    original_dict = {"alpha": "1", "beta": "two", "Gamma": "3.14"}
    opts = Options(original_dict)

    serialized = opts.serialize()
    deserialized = Options.deserialize(serialized)

    restored = deserialized.get_strings()
    assert restored == {k.lower(): v for k, v in original_dict.items()}


def test_serialize_empty_options() -> None:
    opts = Options({})
    serialized = opts.serialize()
    assert isinstance(serialized, bytes)
    assert len(serialized) == 8  # Only the count (0) as uint64_t.

    deserialized = Options.deserialize(serialized)
    assert deserialized.get_strings() == {}


def test_deserialize_invalid_size() -> None:
    with pytest.raises(ValueError):
        Options.deserialize(
            b"\x01\x02\x03\x04\x05\x06\x07"
        )  # 7 bytes (not multiple of 8).


def test_deserialize_odd_offset_count() -> None:
    # Count = 1, but only 1 offset instead of 2 (incomplete pair).
    bad_data = (1).to_bytes(8, "little") + (42).to_bytes(8, "little")
    with pytest.raises(ValueError):
        Options.deserialize(bad_data)


def test_deserialize_out_of_bounds_offset() -> None:
    # Create valid serialized data and tamper with an offset.
    opts = Options({"hello": "world"})
    buf = bytearray(opts.serialize())

    # Overwrite offset to something out of bounds.
    bad_offset = len(buf) + 100
    buf[8:16] = bad_offset.to_bytes(8, "little")  # tamper key offset.
    with pytest.raises(IndexError):
        Options.deserialize(bytes(buf))


def test_serialize_after_access_raises() -> None:
    opts = Options({"x": "42"})
    _ = opts.get_or_assign("x", int, 1)  # Access value.

    with pytest.raises(ValueError):
        _ = opts.serialize()


def test_pickle_roundtrip() -> None:
    original_dict = {"x": "42", "y": "test", "Z": "true"}
    opts = Options(original_dict)

    pickled = pickle.dumps(opts)
    assert isinstance(pickled, bytes)
    unpickled = pickle.loads(pickled)

    assert isinstance(unpickled, Options)
    assert unpickled.get_strings() == {k.lower(): v for k, v in original_dict.items()}


def test_pickle_empty_options() -> None:
    opts = Options({})
    pickled = pickle.dumps(opts)
    unpickled = pickle.loads(pickled)

    assert isinstance(unpickled, Options)
    assert unpickled.get_strings() == {}
