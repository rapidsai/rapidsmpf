#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "duckdb",
#     "numpy",
#     "pyarrow",
#     "tpchgen-cli",
# ]
# ///

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for NDSH benchmarks.

This script validates the correctness of NDSH benchmark outputs by:
1. Running SQL queries via DuckDB to generate expected results
2. Running the C++ benchmark binaries
3. Comparing the benchmark output against the DuckDB result

Usage:
    python validate.py \
        --benchmark-dir /path/to/build/benchmarks/ndsh \
        --sql-dir /path/to/sql/queries \
        --input-dir /raid/rapidsmpf/data/tpch/scale-1.0 \
        --output-dir /tmp/validation
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TPCH_TABLES = [
    "customer",
    "lineitem",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
]


def discover_benchmarks(
    benchmark_dir: Path, sql_dir: Path
) -> list[tuple[str, Path, Path]]:
    """
    Discover benchmark binaries and their corresponding SQL files.

    Returns a list of tuples: (query_name, binary_path, sql_path)
    """
    benchmarks = []
    pattern = re.compile(r"^q(\d+)$")

    for binary in benchmark_dir.iterdir():
        if not binary.is_file():
            continue
        match = pattern.match(binary.name)
        if not match:
            continue

        query_name = binary.name
        sql_path = sql_dir / f"{query_name}.sql"

        if sql_path.exists():
            benchmarks.append((query_name, binary, sql_path))
        else:
            print(f"Warning: No SQL file found for {query_name} at {sql_path}")

    return sorted(benchmarks)


def generate_expected(sql_path: Path, input_dir: Path, output_path: Path) -> None:
    """
    Generate expected results by running a SQL query via DuckDB.

    Parameters
    ----------
    sql_path
        Path to the SQL query file
    input_dir:
        Directory containing TPC-H parquet files
    output_path
        Path to write the expected parquet result
    """
    con = duckdb.connect()

    # Register TPC-H tables as views from parquet files
    for table in TPCH_TABLES:
        # Try both single file and directory patterns
        single_file = input_dir / f"{table}.parquet"
        directory = input_dir / table

        if single_file.exists():
            parquet_path = single_file
        elif directory.exists() and directory.is_dir():
            parquet_path = directory / "*.parquet"
        else:
            raise FileNotFoundError(f"Table {table} not found in {input_dir}")

        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{parquet_path}')"
        )

    # Read and execute the query
    query = sql_path.read_text()
    result = con.execute(query).arrow().read_all()

    # Write result to parquet
    pq.write_table(result, output_path)
    print(f"  Generated expected: {output_path} ({result.num_rows} rows)")


def generate_data(input_dir: Path) -> None:
    """
    Generate data for the benchmarks.

    This uses tpchgen-cli to generate the data and casts some columns
    to the types expected by the benchmarks.
    """
    print(f"Generating data for {input_dir}...")
    subprocess.check_output(
        [
            "tpchgen-cli",
            "--scale-factor",
            "1",
            "--format",
            "parquet",
            "--output-dir",
            str(input_dir),
        ]
    )

    # Some of our queries are written expecting float (Double)
    casts = {
        ("customer", "c_nationkey"): pa.int32(),
        ("customer", "c_acctbal"): pa.float64(),
        ("lineitem", "l_linenumber"): pa.int64(),
        ("lineitem", "l_quantity"): pa.float64(),
        ("lineitem", "l_extendedprice"): pa.float64(),
        ("lineitem", "l_discount"): pa.float64(),
        ("lineitem", "l_tax"): pa.float64(),
        ("lineitem", "l_shipdate"): pa.timestamp("ms"),
        ("lineitem", "l_commitdate"): pa.timestamp("ms"),
        ("lineitem", "l_receiptdate"): pa.timestamp("ms"),
        ("nation", "n_nationkey"): pa.int32(),
        ("nation", "n_regionkey"): pa.int32(),
        ("orders", "o_totalprice"): pa.float64(),
        ("orders", "o_orderdate"): pa.timestamp("ms"),
        ("part", "p_retailprice"): pa.float64(),
        ("partsupp", "ps_availqty"): pa.int64(),
        ("partsupp", "ps_supplycost"): pa.float64(),
        ("region", "r_regionkey"): pa.int32(),
        ("supplier", "s_nationkey"): pa.int32(),
        ("supplier", "s_acctbal"): pa.float64(),
    }

    for table_name in TPCH_TABLES:
        file = (input_dir / table_name).with_suffix(".parquet")
        table = pq.read_table(file)
        schema = table.schema
        for i, field in enumerate(schema):
            if cast := casts.get((table_name, field.name)):
                schema = schema.set(i, field.with_type(cast))

        pq.write_table(table.cast(schema), file)


def run_benchmark(
    binary_path: Path,
    input_dir: Path,
    output_path: Path,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """
    Run a benchmark binary.

    Parameters
    ----------
    binary_path
        Path to the benchmark binary
    input_dir
        Directory containing TPC-H parquet files
    output_path
        Path for benchmark output
    extra_args
        Additional arguments to pass to the benchmark

    Returns
    -------
    CompletedProcess result
    """
    cmd = [
        "mpirun",
        "-np",
        "1",
        "--allow-run-as-root",
        str(binary_path),
        "--input-directory",
        str(input_dir),
        "--output-file",
        str(output_path),
    ]

    if extra_args:
        cmd.extend(extra_args)

    print(f"  Running: {' '.join(cmd)}")

    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )


def compare_parquet(
    output_path: Path,
    expected_path: Path,
    decimal: int = 2,
    *,
    check_timezone: bool = False,
) -> tuple[bool, str | None]:
    """
    Compare two parquet files for exact equality.

    Parameters
    ----------
    output_path
        Path to the benchmark output parquet
    expected_path
        Path to the expected parquet
    decimal
        Number of decimal places to compare for floating point values
    check_timezone
        Whether to check for timezone differences

    Returns
    -------
    Tuple of boolean indicating success and list of error messages. A non-empty list indicates failure.
    """
    try:
        output = pq.read_table(output_path)
        expected = pq.read_table(expected_path)
    except Exception as e:
        return False, f"Failed to read parquet files: {e}"

    # Check schema
    # names
    if output.schema.names != expected.schema.names:
        return (
            False,
            f"Schema name mismatch: {output.schema.names} != {expected.schema.names}",
        )

    # types
    errors = []
    for name in output.schema.names:
        o_field = output.schema.field(name)
        e_field = expected.schema.field(name)
        # We only care about the type, not the metadata or nullability
        if not o_field.type.equals(e_field.type):
            # Ignore differences in timezone and precision
            if (
                not check_timezone
                and pa.types.is_timestamp(o_field.type)
                and pa.types.is_timestamp(e_field.type)
            ):
                continue
            errors.append(f"\t{o_field.type} != {e_field.type}")
    if errors:
        return False, "\n".join(["Field type mismatch (output != expected)", *errors])

    # row count
    if output.num_rows != expected.num_rows:
        return False, (
            f"Row count mismatch: output={output.num_rows}, expected={expected.num_rows}"
        )

    # values. For float types, we'll use approximate equality.
    for name, out_col, expected_col in zip(
        output.column_names, output.columns, expected.columns, strict=False
    ):
        if pa.types.is_floating(out_col.type):
            try:
                np.testing.assert_array_almost_equal(
                    out_col.to_numpy(), expected_col.to_numpy(), decimal=decimal
                )
            except AssertionError as e:
                errors.append(f"{name} differs. {e}")
        else:
            try:
                np.testing.assert_array_equal(
                    out_col.to_numpy(), expected_col.to_numpy()
                )
            except AssertionError as e:
                errors.append(f"{name} differs. {e}")

    if errors:
        return False, "\n".join(errors)

    return True, None


def validate_benchmark(
    query_name: str,
    binary_path: Path,
    sql_path: Path,
    input_dir: Path,
    output_dir: Path,
    extra_args: list[str] | None = None,
    decimal: int = 2,
    *,
    reuse_expected: bool = False,
    reuse_output: bool = False,
) -> bool:
    """
    Validate a single benchmark.

    Returns True if validation passes, False otherwise.
    """
    print(f"\nValidating {query_name}...")

    expected_path = output_dir / f"{query_name}_expected.parquet"
    benchmark_output = output_dir / f"{query_name}_output.parquet"

    # Generate expected
    if reuse_expected and expected_path.exists():
        print(f"  Reusing existing expected: {expected_path}")
    else:
        print("  Generating expected via DuckDB...")
        try:
            generate_expected(sql_path, input_dir, expected_path)
        except Exception as e:
            print(f"  FAILED: Expected generation error: {e}")
            return False

    # Run benchmark
    if reuse_output and benchmark_output.exists():
        print(f"  Reusing existing output: {benchmark_output}")
    else:
        result = run_benchmark(binary_path, input_dir, benchmark_output, extra_args)

        if result.returncode != 0:
            print(f"  FAILED: Benchmark exited with code {result.returncode}")
            print(f"  stdout: {result.stdout[:1000] if result.stdout else '(empty)'}")
            print(f"  stderr: {result.stderr[:1000] if result.stderr else '(empty)'}")
            return False

    if not benchmark_output.exists():
        print(f"  FAILED: Benchmark did not produce output file: {benchmark_output}")
        return False

    # Compare results
    print("  Comparing results...")
    is_equal, message = compare_parquet(
        benchmark_output, expected_path, decimal=decimal
    )

    if is_equal:
        print("  PASSED")
        return True
    else:
        print(f"  FAILED:\n{message}")
        return False


def main():
    """Run the validator."""
    parser = argparse.ArgumentParser(
        description="Validate NDSH benchmarks against DuckDB expected results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        help="Directory containing benchmark binaries (q04, q09, etc.)",
        default=Path(__file__).parent.parent.parent.joinpath(
            "cpp/build/benchmarks/ndsh"
        ),
    )
    parser.add_argument(
        "--sql-dir",
        type=Path,
        help="Directory containing SQL query files (q04.sql, q09.sql, etc.)",
        default=Path(__file__).parent.parent.parent.joinpath(
            "cpp/benchmarks/streaming/ndsh/sql"
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing TPC-H input parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files (default: temp directory)",
    )
    parser.add_argument(
        "--query",
        type=str,
        action="append",
        dest="queries",
        help="Specific query to validate (can be repeated). If not specified, all discovered queries are validated.",
    )
    parser.add_argument(
        "--benchmark-args",
        type=str,
        default="",
        help="Additional arguments to pass to benchmark binaries (space-separated)",
    )
    parser.add_argument(
        "-d",
        "--decimal",
        type=int,
        default=2,
        help="Number of decimal places to compare for floating point values (default: 2)",
    )
    parser.add_argument(
        "--reuse-expected",
        action="store_true",
        help="Skip generating expected results if the expected file already exists",
    )
    parser.add_argument(
        "--reuse-output",
        action="store_true",
        help="Skip running the benchmark if the output file already exists",
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate data for the benchmarks",
    )
    args = parser.parse_args()

    # Validate paths
    if not args.benchmark_dir.exists():
        print(f"Error: Benchmark directory does not exist: {args.benchmark_dir}")
        sys.exit(1)

    if not args.sql_dir.exists():
        print(f"Error: SQL directory does not exist: {args.sql_dir}")
        sys.exit(1)

    if args.generate_data:
        generate_data(args.input_dir)

    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Use temp directory if output dir not specified
    if args.output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="ndsh_validate_"))
        print(f"Using temporary output directory: {output_dir}")
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    # Parse extra benchmark args
    extra_args = args.benchmark_args.split() if args.benchmark_args else None

    # Discover benchmarks
    benchmarks = discover_benchmarks(args.benchmark_dir, args.sql_dir)

    if not benchmarks:
        print("No benchmarks found!")
        sys.exit(1)

    # Filter to specific queries if requested
    if args.queries:
        benchmarks = [
            (name, binary, sql)
            for name, binary, sql in benchmarks
            if name in args.queries
        ]
        if not benchmarks:
            print(f"No matching benchmarks found for queries: {args.queries}")
            sys.exit(1)

    print(f"Found {len(benchmarks)} benchmark(s) to validate:")
    for name, binary, sql in benchmarks:
        print(f"  {name}: {binary} + {sql}")

    # Run validations
    results = {}
    for query_name, binary_path, sql_path in benchmarks:
        passed = validate_benchmark(
            query_name,
            binary_path,
            sql_path,
            args.input_dir,
            output_dir,
            extra_args,
            args.decimal,
            reuse_expected=args.reuse_expected,
            reuse_output=args.reuse_output,
        )
        results[query_name] = passed

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    failed = len(results) - passed

    for query_name, result in sorted(results.items()):
        status = "PASSED" if result else "FAILED"
        print(f"  {query_name}: {status}")

    print("-" * 60)
    print(f"Total: {passed} passed, {failed} failed")

    sys.exit(int(failed > 0))


if __name__ == "__main__":
    main()
