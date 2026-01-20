#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate topology_discovery JSON output for correctness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_json_from_path_or_stdin(input_path: str) -> Any:
    """
    Load JSON from a file path or stdin.

    Parameters
    ----------
    input_path
        Path to JSON file, or "-" to read from stdin.

    Returns
    -------
    Parsed JSON data.
    """
    if input_path == "-":
        try:
            content = sys.stdin.read()
        except Exception as exc:
            raise RuntimeError(f"failed reading stdin: {exc}") from exc
        if not content.strip():
            raise ValueError(
                "no input provided on stdin; pass --input <file> or pipe JSON"
            )
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON from stdin: {exc}") from exc
    else:
        try:
            with Path(input_path).open(encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"input file not found: {input_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in file {input_path}: {exc}") from exc


def ensure_non_empty_string(
    errors: list[str], obj: Any, key: str, context: str
) -> None:
    """Validate that a key contains a non-empty string value."""
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{context}.{key} must be a non-empty string")


def ensure_int(
    errors: list[str], obj: Any, key: str, context: str, *, min_value: int | None = None
) -> None:
    """Validate that a key contains an integer value, optionally with a minimum."""
    value = obj.get(key)
    if not isinstance(value, int):
        errors.append(f"{context}.{key} must be an integer")
        return
    if min_value is not None and value < min_value:
        errors.append(f"{context}.{key} must be >= {min_value} (got {value})")


def ensure_non_empty_list(errors: list[str], obj: Any, key: str, context: str) -> None:
    """Validate that a key contains a non-empty list."""
    value = obj.get(key)
    if not isinstance(value, list) or len(value) == 0:
        errors.append(f"{context}.{key} must be a non-empty list")


def validate_topology(data: Any) -> list[str]:
    """Validate topology JSON data structure and return list of errors."""
    errors: list[str] = []

    if not isinstance(data, dict):
        return ["top-level JSON must be an object"]

    # System section
    system = data.get("system")
    if not isinstance(system, dict):
        errors.append("system must be an object")
    else:
        ensure_non_empty_string(errors, system, "hostname", "system")
        ensure_int(errors, system, "num_gpus", "system", min_value=1)
        ensure_int(errors, system, "num_numa_nodes", "system", min_value=1)
        ensure_int(errors, system, "num_network_devices", "system", min_value=0)

    # GPUs section
    gpus = data.get("gpus")
    if not isinstance(gpus, list) or len(gpus) == 0:
        errors.append(
            "gpus must be a non-empty array (at least one GPU must be present)"
        )
    else:
        gpu_ids = []
        for index, gpu in enumerate(gpus):
            ctx = f"gpus[{index}]"
            if not isinstance(gpu, dict):
                errors.append(f"{ctx} must be an object")
                continue

            ensure_int(errors, gpu, "id", ctx)
            if isinstance(gpu.get("id"), int):
                gpu_ids.append(gpu["id"])
            ensure_non_empty_string(errors, gpu, "name", ctx)
            ensure_non_empty_string(errors, gpu, "pci_bus_id", ctx)
            ensure_non_empty_string(errors, gpu, "uuid", ctx)
            ensure_int(errors, gpu, "numa_node", ctx)

            cpu_affinity = gpu.get("cpu_affinity")
            if not isinstance(cpu_affinity, dict):
                errors.append(f"{ctx}.cpu_affinity must be an object")
            else:
                ensure_non_empty_string(
                    errors, cpu_affinity, "cpulist", f"{ctx}.cpu_affinity"
                )
                ensure_non_empty_list(
                    errors, cpu_affinity, "cores", f"{ctx}.cpu_affinity"
                )

        # GPU id uniqueness check
        if len(gpu_ids) != len(set(gpu_ids)):
            seen = set()
            dups = set()
            for gid in gpu_ids:
                if gid in seen:
                    dups.add(gid)
                else:
                    seen.add(gid)
            dup_list = ", ".join(str(x) for x in sorted(dups))
            errors.append(f"gpus ids must be unique; duplicates found: {dup_list}")

    # We don't validate several fields and their contents because the virtualized CI
    # environment doesn't have complete topology information.

    return errors


def main() -> int:
    """Entry point for the topology JSON validator CLI."""
    parser = argparse.ArgumentParser(
        description="Validate topology_discovery JSON output"
    )
    parser.add_argument(
        "input",
        help="Path to JSON file to validate; pass '-' to read from stdin",
    )
    args = parser.parse_args()

    try:
        data = load_json_from_path_or_stdin(args.input)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    errors = validate_topology(data)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
