#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyze nsys reports for postbox spilling statistics."""

from __future__ import annotations

import argparse
import math
import sqlite3
import statistics
import subprocess
import sys
import textwrap
from pathlib import Path


def export_nsys_rep(nsys_rep_path: Path, *, force_overwrite: bool) -> Path:
    """Export .nsys-rep to .sqlite using nsys export."""
    sqlite_path = nsys_rep_path.with_suffix(".sqlite")

    cmd = ["nsys", "export", "-t", "sqlite", str(nsys_rep_path)]
    if force_overwrite:
        cmd.extend(["--force-overwrite", "true"])

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error exporting nsys report: {e.stderr}", file=sys.stderr)
        raise
    except FileNotFoundError:
        print(
            "Error: 'nsys' command not found. Please ensure NVIDIA Nsight Systems is installed.",
            file=sys.stderr,
        )

    return sqlite_path


def format_bytes(n: float) -> str:
    """Format bytes in human-readable form."""
    if n == 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = min(int(math.log2(abs(n)) / 10), len(units) - 1)
    return f"{n / (1024**i):.2f} {units[i]}"


def format_duration(ns: float) -> str:
    """Format nanoseconds in human-readable form."""
    if ns < 1000:
        return f"{ns:.2f} ns"
    elif ns < 1_000_000:
        return f"{ns / 1000:.2f} µs"
    elif ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    else:
        return f"{ns / 1_000_000_000:.2f} s"


def compute_stats(values: list[float]) -> dict[str, float]:
    """Compute avg, std, min, max for a list of values."""
    if not values:
        return {"avg": 0, "std": 0, "min": 0, "max": 0}

    avg = statistics.mean(values)
    std = statistics.stdev(values)

    return {
        "avg": avg,
        "std": std,
        "min": min(values),
        "max": max(values),
    }


def analyze_spilling(sqlite_path: Path) -> None:
    """Analyze postbox spilling from nsys SQLite export."""
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Check if required tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}

    if "NVTX_EVENTS" not in tables:
        print("Error: NVTX_EVENTS table not found in SQLite file.", file=sys.stderr)
        print(
            "Make sure the nsys profile was captured with NVTX tracing enabled.",
            file=sys.stderr,
        )

    # Query for postbox_spilling ranges (duration events)
    # These have the "amount requested" as payload
    # Check both: text column (unregistered) and StringIds (registered)
    cursor.execute("""
        SELECT
            e.start,
            e.end,
            COALESCE(e.int64Value, e.uint64Value, e.int32Value, e.uint32Value,
                     e.doubleValue, e.floatValue) as payload
        FROM NVTX_EVENTS e
        WHERE e.text = 'postbox_spilling'
          AND e.end IS NOT NULL
          AND e.end > 0
        UNION ALL
        SELECT
            e.start,
            e.end,
            COALESCE(e.int64Value, e.uint64Value, e.int32Value, e.uint32Value,
                     e.doubleValue, e.floatValue) as payload
        FROM NVTX_EVENTS e
        JOIN StringIds s ON e.textId = s.id
        WHERE s.value = 'postbox_spilling'
          AND e.end IS NOT NULL
          AND e.end > 0
    """)
    spill_ranges = cursor.fetchall()

    # Query for postbox_spilling::total_spilled markers
    # These have the "actual bytes spilled" as payload
    # Check both: text column (unregistered) and StringIds (registered)
    cursor.execute("""
        SELECT
            e.start,
            COALESCE(e.int64Value, e.uint64Value, e.int32Value, e.uint32Value,
                     e.doubleValue, e.floatValue) as payload
        FROM NVTX_EVENTS e
        WHERE e.text = 'postbox_spilling::total_spilled'
        UNION ALL
        SELECT
            e.start,
            COALESCE(e.int64Value, e.uint64Value, e.int32Value, e.uint32Value,
                     e.doubleValue, e.floatValue) as payload
        FROM NVTX_EVENTS e
        JOIN StringIds s ON e.textId = s.id
        WHERE s.value = 'postbox_spilling::total_spilled'
    """)
    spill_markers = cursor.fetchall()

    conn.close()

    # Extract data
    durations = [(end - start) for start, end, _ in spill_ranges]
    bytes_requested = [payload for _, _, payload in spill_ranges if payload is not None]
    bytes_spilled = [payload for _, payload in spill_markers if payload is not None]

    # Compute statistics
    duration_stats = compute_stats(durations)
    requested_stats = compute_stats(bytes_requested)
    spilled_stats = compute_stats(bytes_spilled)

    # Summary totals
    total_requested = sum(bytes_requested) if bytes_requested else 0
    total_spilled = sum(bytes_spilled) if bytes_spilled else 0
    total_duration = sum(durations) if durations else 0

    # Print report
    print("Postbox Spilling Summary")
    print("=" * 80)
    print()

    call_count = len(spill_ranges)

    if call_count == 0:
        print("No postbox_spilling events found.")
        return

    # Build table data
    rows = [
        (
            "Duration",
            format_duration(duration_stats["avg"]),
            format_duration(duration_stats["std"]),
            format_duration(duration_stats["min"]),
            format_duration(duration_stats["max"]),
            format_duration(total_duration),
        ),
        (
            "Bytes requested",
            format_bytes(requested_stats["avg"]),
            format_bytes(requested_stats["std"]),
            format_bytes(requested_stats["min"]),
            format_bytes(requested_stats["max"]),
            format_bytes(total_requested),
        ),
        (
            "Bytes spilled",
            format_bytes(spilled_stats["avg"]),
            format_bytes(spilled_stats["std"]),
            format_bytes(spilled_stats["min"]),
            format_bytes(spilled_stats["max"]),
            format_bytes(total_spilled),
        ),
    ]

    # Column headers
    headers = ["Metric", "Avg", "Std", "Min", "Max", "Total"]

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    # Format row helper
    def format_row(values: tuple) -> str:
        return " | ".join(str(v).rjust(col_widths[i]) for i, v in enumerate(values))

    # Print table
    header_line = format_row(tuple(headers))
    separator = "-+-".join("-" * w for w in col_widths)

    print(header_line)
    print(separator)
    for row in rows:
        print(format_row(row))

    print("")

    print("Spill count: ", call_count)
    if total_requested > 0:
        # What percentage of the spill requests actually satisfy?
        # A satisfaction of 0% means we didn't spill anything.
        # A satisfaction of 100% means we spilled everything we requested.
        efficiency = (total_spilled / total_requested) * 100
        print(f"Spill satisfaction: {efficiency:.1f}%")
    if total_duration > 0:
        # What is the effective throughput (in bytes per second) of
        # the spill requests?
        # Convert nanoseconds to seconds for throughput calculation
        throughput = total_spilled / (total_duration / 1_000_000_000)
        print(f"Effective throughput: {format_bytes(throughput)}/s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze nsys reports for postbox spilling statistics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
    Examples:
      python -m rapidsmpf.report profile.nsys-rep
      python -m rapidsmpf.report profile.sqlite
      python -m rapidsmpf.report profile.nsys-rep --force-overwrite
            """),
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to .nsys-rep or .sqlite file",
    )
    parser.add_argument(
        "-f",
        "--force-overwrite",
        action="store_true",
        help="Overwrite existing .sqlite file when exporting from .nsys-rep",
    )

    args = parser.parse_args()
    input_path = args.input_file

    try:
        if not input_path.exists():
            print(f"Error: File not found: {input_path}", file=sys.stderr)

        # Determine if we need to export
        if input_path.suffix == ".nsys-rep":
            sqlite_path = export_nsys_rep(
                input_path, force_overwrite=args.force_overwrite
            )
        elif input_path.suffix == ".sqlite":
            sqlite_path = input_path
        else:
            print(f"Error: Unsupported file type: {input_path.suffix}", file=sys.stderr)
            print("Expected .nsys-rep or .sqlite file.", file=sys.stderr)
        analyze_spilling(sqlite_path)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
