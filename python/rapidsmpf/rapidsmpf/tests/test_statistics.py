# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import pathlib
import pickle
from typing import TYPE_CHECKING

import pytest

from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.statistics import Formatter, Statistics

if TYPE_CHECKING:
    import rmm.mr


def test_disabled() -> None:
    stats = Statistics(enable=False)
    assert not stats.enabled


def test_add_get_stat() -> None:
    stats = Statistics(enable=True)
    assert stats.enabled

    stats.add_stat("stat1", 1)
    stats.add_stat("stat2", 2)
    stats.add_stat("stat1", 4)
    assert stats.get_stat("stat1") == {"count": 2, "value": 5.0, "max": 4.0}
    assert stats.get_stat("stat2") == {"count": 1, "value": 2, "max": 2.0}


def test_get_nonexistent_stat() -> None:
    """Test that accessing a non-existent statistic raises KeyError."""
    stats = Statistics(enable=True)

    with pytest.raises(KeyError, match="Statistic 'foo' does not exist"):
        stats.get_stat("foo")


def test_enable_memory_profiling(device_mr: rmm.mr.CudaMemoryResource) -> None:
    stats = Statistics(enable=False)
    assert not stats.memory_profiling_enabled
    mr = RmmResourceAdaptor(device_mr)
    stats = Statistics(enable=False, mr=mr)
    assert not stats.memory_profiling_enabled
    stats = Statistics(enable=True, mr=mr)
    assert stats.memory_profiling_enabled


def test_get_empty_memory_records(device_mr: rmm.mr.CudaMemoryResource) -> None:
    stats = Statistics(enable=True, mr=RmmResourceAdaptor(device_mr))
    assert stats.get_memory_records() == {}


def test_memory_profiling(device_mr: rmm.mr.CudaMemoryResource) -> None:
    mr = RmmResourceAdaptor(device_mr)
    stats = Statistics(enable=True, mr=mr)
    with stats.memory_profiling("outer"):
        b1 = mr.allocate(1024)
        with stats.memory_profiling("inner"):
            mr.deallocate(mr.allocate(512), 512)
            mr.deallocate(mr.allocate(512), 512)
        mr.deallocate(b1, 1024)
        mr.deallocate(mr.allocate(1024), 1024)

    inner = stats.get_memory_records()["inner"]
    assert inner.scoped.num_total_allocs() == 2
    assert inner.scoped.peak() == 512
    assert inner.scoped.total() == 512 + 512
    assert inner.global_peak == 512
    assert inner.num_calls == 1

    outer = stats.get_memory_records()["outer"]
    assert outer.scoped.num_total_allocs() == 2 + 2
    assert outer.scoped.peak() == 1024 + 512
    assert outer.scoped.total() == 1024 + 1024 + 512 + 512
    assert outer.global_peak == 1024 + 512
    assert outer.num_calls == 1


def test_list_stat_names() -> None:
    stats = Statistics(enable=True)
    assert stats.list_stat_names() == []
    stats.add_stat("stat1", 1.0)
    assert stats.list_stat_names() == ["stat1"]
    stats.add_stat("stat2", 2.0)
    assert stats.list_stat_names() == ["stat1", "stat2"]
    stats.add_stat("stat1", 3.0)
    assert stats.list_stat_names() == ["stat1", "stat2"]


def test_clear() -> None:
    stats = Statistics(enable=True)
    stats.add_stat("stat1", 10.0)

    assert stats.get_stat("stat1") == {"count": 1, "value": 10.0, "max": 10.0}
    stats.clear()
    assert stats.list_stat_names() == []

    stats.add_stat("stat1", 10.0)
    assert stats.get_stat("stat1") == {"count": 1, "value": 10.0, "max": 10.0}


def test_write_json_string() -> None:
    stats = Statistics(enable=True)
    stats.add_stat("foo", 10.0)
    stats.add_stat("foo", 5.0)

    data = json.loads(stats.write_json_string())
    assert data["statistics"]["foo"]["count"] == 2
    assert data["statistics"]["foo"]["value"] == 15.0
    assert data["statistics"]["foo"]["max"] == 10.0
    assert "memory_records" not in data


def test_write_json_string_matches_file(tmp_path: pathlib.Path) -> None:
    stats = Statistics(enable=True)
    stats.add_stat("foo", 10.0)
    stats.add_stat("foo", 5.0)

    out = tmp_path / "stats.json"
    stats.write_json(out)
    assert out.read_text() == stats.write_json_string()


def test_write_json_memory_records(device_mr: rmm.mr.CudaMemoryResource) -> None:
    mr = RmmResourceAdaptor(device_mr)
    stats = Statistics(enable=True, mr=mr)
    with stats.memory_profiling("alloc"):
        mr.deallocate(mr.allocate(1024), 1024)

    data = json.loads(stats.write_json_string())
    assert "memory_records" in data
    rec = data["memory_records"]["alloc"]
    assert rec["num_calls"] == 1
    assert rec["peak_bytes"] > 0
    assert rec["total_bytes"] > 0
    assert rec["global_peak_bytes"] > 0


def test_invalid_memory_record_names(device_mr: rmm.mr.CudaMemoryResource) -> None:
    mr = RmmResourceAdaptor(device_mr)
    stats = Statistics(enable=True, mr=mr)
    with stats.memory_profiling('bad"name'):
        pass
    with pytest.raises(ValueError):
        stats.write_json_string()


def test_invalid_stat_names() -> None:
    stats = Statistics(enable=True)
    stats.add_stat('has"quote', 1.0)
    stats.add_stat("has\\backslash", 2.0)
    with pytest.raises(ValueError):
        stats.write_json_string()


@pytest.mark.parametrize("as_type", [pathlib.Path, str], ids=["pathlib.Path", "str"])
def test_write_json(tmp_path: pathlib.Path, as_type: type) -> None:
    stats = Statistics(enable=True)
    stats.add_stat("foo", 10.0)
    stats.add_stat("foo", 5.0)

    out = tmp_path / "stats.json"
    stats.write_json(as_type(out))

    data = json.loads(out.read_text())
    assert data["statistics"]["foo"]["count"] == 2
    assert data["statistics"]["foo"]["value"] == 15.0
    assert data["statistics"]["foo"]["max"] == 10.0
    assert "memory_records" not in data


def test_copy() -> None:
    stats = Statistics(enable=True)
    stats.add_stat("x", 10.0)

    copied = stats.copy()
    assert copied.enabled
    assert copied.get_stat("x") == stats.get_stat("x")

    # Modifying the copy does not affect the original.
    copied.add_stat("y", 1.0)
    assert "y" not in stats.list_stat_names()


def test_merge_overlapping() -> None:
    a = Statistics(enable=True)
    a.add_stat("x", 10.0)
    a.add_stat("x", 3.0)

    b = Statistics(enable=True)
    b.add_stat("x", 7.0)

    merged = Statistics.merge([a, b])
    s = merged.get_stat("x")
    assert s["count"] == 3
    assert s["value"] == 20.0
    assert s["max"] == 10.0


def test_merge_disjoint() -> None:
    a = Statistics(enable=True)
    a.add_stat("x", 1.0)

    b = Statistics(enable=True)
    b.add_stat("y", 2.0)

    merged = Statistics.merge([a, b])
    assert len(merged.list_stat_names()) == 2
    assert merged.get_stat("x") == a.get_stat("x")
    assert merged.get_stat("y") == b.get_stat("y")


def test_merge_multiple() -> None:
    a = Statistics(enable=True)
    a.add_stat("x", 1.0)

    b = Statistics(enable=True)
    b.add_stat("x", 2.0)

    c = Statistics(enable=True)
    c.add_stat("y", 10.0)

    merged = Statistics.merge([a, b, c])
    assert len(merged.list_stat_names()) == 2
    assert merged.get_stat("x")["value"] == 3.0
    assert merged.get_stat("y")["value"] == 10.0


def test_merge_rejects_empty() -> None:
    with pytest.raises(ValueError):
        Statistics.merge([])


def test_merge_rejects_conflicting_formatter() -> None:
    a = Statistics(enable=True)
    a.add_report_entry("x", ["x"], Formatter.Bytes)
    a.add_stat("x", 1.0)

    b = Statistics(enable=True)
    b.add_report_entry("x", ["x"], Formatter.Duration)
    b.add_stat("x", 2.0)

    with pytest.raises(ValueError):
        Statistics.merge([a, b])


def test_merge_rejects_conflicting_stat_names() -> None:
    a = Statistics(enable=True)
    a.add_report_entry("copy", ["b1", "t1", "d1"], Formatter.MemoryThroughput)

    b = Statistics(enable=True)
    b.add_report_entry("copy", ["b2", "t2", "d2"], Formatter.MemoryThroughput)

    with pytest.raises(ValueError):
        Statistics.merge([a, b])


def test_merge_preserves_report_entry_formatter() -> None:
    # Both sides declare the same Bytes report entry; the merged report
    # renders with the Bytes formatter.
    a = Statistics(enable=True)
    a.add_report_entry("payload", ["payload"], Formatter.Bytes)
    a.add_stat("payload", 1024.0)

    b = Statistics(enable=True)
    b.add_report_entry("payload", ["payload"], Formatter.Bytes)
    b.add_stat("payload", 1024.0)

    merged = Statistics.merge([a, b])
    assert "2 KiB" in merged.report()


def test_add_report_entry_bytes() -> None:
    stats = Statistics(enable=True)
    stats.add_stat("payload", 1024.0)
    stats.add_report_entry("payload", ["payload"], Formatter.Bytes)
    assert "1 KiB" in stats.report()


def test_add_report_entry_memcopy() -> None:
    stats = Statistics(enable=True)
    stats.add_stat("copy-bytes", 1024 * 1024)
    stats.add_stat("copy-time", 0.002)
    stats.add_stat("copy-delay", 0.00001)
    stats.add_report_entry(
        "copy", ["copy-bytes", "copy-time", "copy-delay"], Formatter.MemoryThroughput
    )
    assert "1 MiB" in stats.report()


def test_json_has_no_formatter_info() -> None:
    stats = Statistics(enable=True)
    stats.add_stat("alpha", 1024.0)
    stats.add_report_entry("alpha", ["alpha"], Formatter.Bytes)

    data = json.loads(stats.write_json_string())
    # JSON carries numeric data only — no formatter/report-entry metadata.
    assert "report_entries" not in data
    assert "formatter" not in data


def test_pickle_roundtrip() -> None:
    stats = Statistics(enable=True)
    stats.add_stat("payload", 2048.0)
    stats.add_report_entry("payload", ["payload"], Formatter.Bytes)
    stats.add_stat("plain", 7.0)
    stats.add_report_entry(
        "copy",
        ["copy-bytes", "copy-time", "copy-delay"],
        Formatter.MemoryThroughput,
    )
    stats.add_stat("copy-bytes", 1024 * 1024)
    stats.add_stat("copy-time", 0.002)
    stats.add_stat("copy-delay", 0.00001)

    unpickled = pickle.loads(pickle.dumps(stats))
    assert isinstance(unpickled, Statistics)
    # Stats round-trip numerically.
    assert unpickled.get_stat("payload") == stats.get_stat("payload")
    assert unpickled.get_stat("plain") == stats.get_stat("plain")
    # Formatter metadata round-trips — the report is identical.
    assert unpickled.report() == stats.report()


def test_pickle_empty() -> None:
    stats = Statistics(enable=True)
    unpickled = pickle.loads(pickle.dumps(stats))
    assert unpickled.enabled
    assert unpickled.list_stat_names() == []


def test_pickle_preserves_disabled_flag() -> None:
    stats = Statistics(enable=False)
    assert not stats.enabled
    unpickled = pickle.loads(pickle.dumps(stats))
    assert not unpickled.enabled
