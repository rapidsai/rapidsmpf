# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pickle

import pytest

from rapidsmpf.config import Options


def test_get_with_explicit_values() -> None:
    opts = Options(
        {
            "debug": "true",
            "max_retries": "3",
            "timeout": "2.5",
            "int": "2.5",
            "mode": "fast",
        }
    )
    assert opts.get("debug", return_type=bool, factory=lambda s: s == "true") is True
    assert opts.get("max_retries", return_type=int, factory=int) == 3
    assert opts.get("timeout", return_type=float, factory=float) == 2.5
    assert opts.get("mode", return_type=str, factory=str) == "fast"


def test_get_uses_factory_when_key_missing() -> None:
    opts = Options({})
    assert opts.get("use_gpu", return_type=bool, factory=lambda s: True) is True
    assert opts.get("workers", return_type=int, factory=lambda s: 4) == 4
    assert opts.get("rate", return_type=float, factory=lambda s: 1.2) == 1.2
    assert opts.get("name", return_type=str, factory=lambda s: "default") == "default"


def test_get_caches_assigned_value() -> None:
    opts = Options({})

    val1 = opts.get("threshold", return_type=float, factory=lambda s: 0.75)
    val2 = opts.get("threshold", return_type=float, factory=lambda s: 1.23)
    assert val1 == val2 == 0.75  # second call must return the cached value


def test_get_raises_on_unsupported_type() -> None:
    class Unsupported:
        pass

    opts = Options({})
    with pytest.raises(ValueError, match="is not supported"):
        opts.get("key", return_type=Unsupported, factory=lambda s: Unsupported())


def test_get_raises_on_type_conflict() -> None:
    opts = Options({"batch_size": "32"})

    val = opts.get("batch_size", return_type=int, factory=int)
    assert val == 32

    with pytest.raises(ValueError, match="incompatible template type"):
        opts.get("batch_size", return_type=float, factory=float)


def test_get_int64_overflow() -> None:
    opts = Options({"large_int": str(2**65)})

    with pytest.raises(OverflowError, match="too large"):
        opts.get("large_int", return_type=int, factory=int)

    with pytest.raises(OverflowError, match="too large"):
        opts.get("another_large_int", return_type=int, factory=lambda s: 2**65)


def test_get_raises_on_list_type() -> None:
    opts = Options({})
    with pytest.raises(ValueError, match="is not supported"):
        opts.get("some_key", return_type=list, factory=lambda s: [])


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
    _ = opts.get("x", return_type=int, factory=int)  # Access value.

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
