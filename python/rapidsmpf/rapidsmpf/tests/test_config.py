# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gc
import pickle
import weakref
from typing import Any

import pytest

import rmm.mr

from rapidsmpf.config import Optional, OptionalBytes, Options
from rapidsmpf.memory.buffer_resource import (
    AvailableMemoryMap,
    periodic_spill_check_from_options,
    stream_pool_from_options,
)
from rapidsmpf.memory.pinned_memory_resource import (
    PinnedMemoryResource,
    is_pinned_memory_resources_supported,
)
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.statistics import Statistics


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
    opts = Options()
    assert opts.get("use_gpu", return_type=bool, factory=lambda s: True) is True
    assert opts.get("workers", return_type=int, factory=lambda s: 4) == 4
    assert opts.get("rate", return_type=float, factory=lambda s: 1.2) == 1.2
    assert opts.get("name", return_type=str, factory=lambda s: "default") == "default"


def test_get_caches_assigned_value() -> None:
    opts = Options()

    val1 = opts.get("threshold", return_type=float, factory=lambda s: 0.75)
    val2 = opts.get("threshold", return_type=float, factory=lambda s: 1.23)
    assert val1 == val2 == 0.75  # second call must return the cached value


def test_get_accepts_custom_python_type() -> None:
    class Custom:
        def __init__(self, text: str) -> None:
            self.text = text

    opts = Options({"key": "value"})
    result = opts.get("key", return_type=Custom, factory=Custom)
    assert isinstance(result, Custom)
    assert result.text == "value"


def test_get_caches_custom_python_object() -> None:
    class MyObj:
        def __init__(self, s: str) -> None:
            self.s = s

    opts = Options({"thing": "foo"})
    first = opts.get("thing", return_type=MyObj, factory=MyObj)
    second = opts.get("thing", return_type=MyObj, factory=lambda s: MyObj("bar"))
    assert first is second
    assert first.s == "foo"


def test_get_python_object_when_key_missing() -> None:
    class Token:
        def __init__(self, val: str) -> None:
            self.val = val

    opts = Options()
    tok = opts.get("auth", return_type=Token, factory=lambda s: Token("generated"))
    assert isinstance(tok, Token)
    assert tok.val == "generated"


def test_get_list_from_factory() -> None:
    opts = Options({"mylist": "ignored"})
    val = opts.get("mylist", return_type=list, factory=lambda s: [1, 2, 3])
    assert val == [1, 2, 3]


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


def test_get_strings_returns_correct_data() -> None:
    input_data = {"Alpha": "one", "BETA": "2", "gamma": "THREE"}

    opts = Options(input_data)
    result = opts.get_strings()

    # Keys should be lowercase
    expected_keys = {k.lower() for k in input_data}
    assert set(result.keys()) == expected_keys

    for k, v in input_data.items():
        assert result[k.lower()] == v


def test_get_pyobject_refcount() -> None:
    class MyObject:
        def __init__(self, _: str) -> None:
            pass

    opts = Options()
    wr = weakref.ref(opts.get("obj", return_type=MyObject, factory=MyObject))

    # `opts` should keep obj alive.
    assert isinstance(wr(), MyObject)
    del opts
    gc.collect()

    # but without `opts`, no one is keeping obj alive.
    assert wr() is None


def test_get_or_default_returns_default_when_key_missing() -> None:
    opts = Options()
    assert opts.get_or_default("debug", default_value=False) is False
    assert opts.get_or_default("workers", default_value=8) == 8
    assert opts.get_or_default("timeout", default_value=1.5) == 1.5
    assert opts.get_or_default("loglevel", default_value="info") == "info"


def test_get_or_default_parses_existing_values() -> None:
    opts = Options(
        {
            "debug": "true",
            "workers": "4",
            "timeout": "2.0",
            "loglevel": "debug",
        }
    )
    assert opts.get_or_default("debug", default_value=False) is True
    assert opts.get_or_default("workers", default_value=1) == 4
    assert opts.get_or_default("timeout", default_value=1.0) == 2.0
    assert opts.get_or_default("loglevel", default_value="info") == "debug"


def test_get_or_default_type_conflict_raises() -> None:
    opts = Options({"port": "8080"})
    assert opts.get_or_default("port", default_value=8080) == 8080

    with pytest.raises(ValueError, match="incompatible template type"):
        opts.get_or_default("port", default_value=8080.0)


def test_get_or_default_handles_bool_variants() -> None:
    opts = Options({"enabled": "yes", "disabled": "0"})
    assert opts.get_or_default("enabled", default_value=False) is True
    assert opts.get_or_default("disabled", default_value=True) is False


def test_get_or_default_raises_for_invalid_bool_string() -> None:
    opts = Options({"enabled": "definitely"})

    with pytest.raises(ValueError, match="Cannot parse boolean"):
        opts.get_or_default("enabled", default_value=True)


@pytest.mark.parametrize(
    "input_value,expected",
    [
        ("false", None),
        ("FALSE", None),
        ("no", None),
        ("No", None),
        ("off", None),
        ("OFF", None),
        ("disable", None),
        ("DISABLE", None),
        ("disabled", None),
        ("Disabled", None),
        (" true ", " true "),  # not a disable keyword
        ("100", "100"),
        ("", ""),
        (123, 123),  # non-string input preserved
        (None, None),  # None input preserved
        ("  off  ", None),  # whitespace stripped
    ],
)
def test_Optional_values(input_value: Any, expected: Any) -> None:
    d = Optional(input_value)
    assert d.value == expected


def test_Optional_with_options_returns_default_value() -> None:
    opts = Options()
    val = opts.get_or_default("dask_periodic_spill_check", default_value=Optional(42))
    assert isinstance(val, Optional)
    assert val.value == 42


def test_Optional_overrides_with_disabled_string() -> None:
    opts = Options({"dask_periodic_spill_check": "off"})
    val = opts.get_or_default("dask_periodic_spill_check", default_value=Optional(42))
    assert isinstance(val, Optional)
    assert val.value is None


def test_Optional_default_can_be_none() -> None:
    opts = Options()
    val = opts.get_or_default("some_key", default_value=Optional(None))
    assert isinstance(val, Optional)
    assert val.value is None


def test_Optionalbytes_with_options() -> None:
    opts = Options()
    val = opts.get_or_default("max_transfer", default_value=OptionalBytes("1MiB"))
    assert isinstance(val, OptionalBytes)
    assert val.value == 2**20


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


def test_insert_if_absent_inserts_new_keys() -> None:
    opts = Options()
    # Insert 2 new keys
    inserted_count = opts.insert_if_absent({"key1": "1", "key2": "2"})

    assert inserted_count == 2
    assert opts.get("key1", return_type=int, factory=int) == 1
    assert opts.get("key2", return_type=int, factory=int) == 2


def test_insert_if_absent_skips_existing_keys() -> None:
    # Initialize with existing key
    opts = Options({"existing": "old"})
    # Try inserting 1 existing + 1 new key
    inserted_count = opts.insert_if_absent({"existing": "new", "newkey": "value"})

    assert inserted_count == 1
    assert (
        opts.get("existing", return_type=str, factory=str) == "old"
    )  # old value preserved
    assert opts.get("newkey", return_type=str, factory=str) == "value"  # new key added


def test_insert_if_absent_returns_zero_for_empty_input() -> None:
    opts = Options({"existing": "val"})
    # Empty map should insert nothing
    inserted_count = opts.insert_if_absent({})
    assert inserted_count == 0


def test_insert_if_absent_normalizes_keys_before_checking() -> None:
    opts = Options({"lowercase_key": "123"})
    # Try inserting mixed-case and whitespace-padded keys
    inserted_count = opts.insert_if_absent(
        {
            " Lowercase_KEY ": "456",  # matches existing after normalization
            "NEW_KEY": "789",  # new key
        }
    )

    assert inserted_count == 1
    assert (
        opts.get("lowercase_key", return_type=str, factory=str) == "123"
    )  # original preserved
    assert opts.get("new_key", return_type=str, factory=str) == "789"  # new key added


def test_serialize_deserialize_roundtrip() -> None:
    original_dict = {"alpha": "1", "beta": "two", "Gamma": "3.14"}
    opts = Options(original_dict)

    serialized = opts.serialize()
    deserialized = Options.deserialize(serialized)

    restored = deserialized.get_strings()
    assert restored == {k.lower(): v for k, v in original_dict.items()}


def test_serialize_empty_options() -> None:
    opts = Options()
    serialized = opts.serialize()
    assert isinstance(serialized, bytes)
    # v1 format: prelude(8) + count(8) + CRC32(4)
    assert len(serialized) == 8 + 8 + 4

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
    # v1 layout: prelude(8), count(8), key_offset(8), value_offset(8)
    key_offset_start = 8 + 8
    buf[key_offset_start : key_offset_start + 8] = bad_offset.to_bytes(8, "little")
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
    opts = Options()
    pickled = pickle.dumps(opts)
    unpickled = pickle.loads(pickled)
    assert isinstance(unpickled, Options)
    assert unpickled.get_strings() == {}


def test_statistics_from_options_enabled_when_set_to_true() -> None:
    opts = Options({"statistics": "True"})
    mr = RmmResourceAdaptor(rmm.mr.CudaMemoryResource())
    stats = Statistics.from_options(mr, opts)
    assert stats is not None
    assert stats.enabled


def test_statistics_from_options_enabled_when_set_to_one() -> None:
    opts = Options({"statistics": "1"})
    mr = RmmResourceAdaptor(rmm.mr.CudaMemoryResource())
    stats = Statistics.from_options(mr, opts)
    assert stats is not None
    assert stats.enabled


def test_statistics_from_options_disabled_when_set_to_false() -> None:
    opts = Options({"statistics": "False"})
    mr = RmmResourceAdaptor(rmm.mr.CudaMemoryResource())
    stats = Statistics.from_options(mr, opts)
    assert stats is not None
    assert not stats.enabled


def test_statistics_from_options_disabled_by_default() -> None:
    opts = Options()
    mr = RmmResourceAdaptor(rmm.mr.CudaMemoryResource())
    stats = Statistics.from_options(mr, opts)
    assert stats is not None
    assert not stats.enabled


def test_pinned_memory_resource_from_options_enabled_when_set_to_true() -> None:
    opts = Options({"pinned_memory": "True"})
    pmr = PinnedMemoryResource.from_options(opts)

    # Should be enabled if system supports it, or None if not
    if is_pinned_memory_resources_supported():
        assert pmr is not None
    else:
        assert pmr is None


def test_pinned_memory_resource_from_options_disabled_when_set_to_false() -> None:
    opts = Options({"pinned_memory": "False"})
    pmr = PinnedMemoryResource.from_options(opts)
    assert pmr is None


def test_pinned_memory_resource_from_options_disabled_by_default() -> None:
    opts = Options()
    pmr = PinnedMemoryResource.from_options(opts)
    assert pmr is None


def test_available_memory_map_from_options_creates_map() -> None:
    opts = Options({"spill_device_limit": "1GiB"})
    mr = RmmResourceAdaptor(rmm.mr.CudaMemoryResource())
    mem_map = AvailableMemoryMap.from_options(mr, opts)
    assert mem_map is not None


def test_available_memory_map_from_options_uses_default() -> None:
    opts = Options()
    mr = RmmResourceAdaptor(rmm.mr.CudaMemoryResource())
    mem_map = AvailableMemoryMap.from_options(mr, opts)
    assert mem_map is not None


def test_periodic_spill_check_from_options_parses_milliseconds() -> None:
    opts = Options({"periodic_spill_check": "5ms"})
    duration = periodic_spill_check_from_options(opts)
    assert duration is not None
    assert abs(duration - 0.005) < 1e-9  # 5ms = 0.005s


def test_periodic_spill_check_from_options_parses_seconds() -> None:
    opts = Options({"periodic_spill_check": "2"})
    duration = periodic_spill_check_from_options(opts)
    assert duration is not None
    assert abs(duration - 2.0) < 1e-9


def test_periodic_spill_check_from_options_disabled_when_set_to_disabled() -> None:
    opts = Options({"periodic_spill_check": "disabled"})
    duration = periodic_spill_check_from_options(opts)
    assert duration is None


def test_periodic_spill_check_from_options_uses_default() -> None:
    opts = Options()
    duration = periodic_spill_check_from_options(opts)
    assert duration is not None
    assert abs(duration - 0.001) < 1e-9  # Default: 1ms


def test_stream_pool_from_options_returns_specified_size() -> None:
    opts = Options({"num_streams": "32"})
    pool_size = stream_pool_from_options(opts)
    assert pool_size.get_pool_size() == 32


def test_stream_pool_from_options_uses_default() -> None:
    opts = Options()
    pool_size = stream_pool_from_options(opts)
    assert pool_size.get_pool_size() == 16  # Default


def test_stream_pool_from_options_raises_on_zero() -> None:
    opts = Options({"num_streams": "0"})
    with pytest.raises(ValueError, match="greater than 0"):
        stream_pool_from_options(opts)
