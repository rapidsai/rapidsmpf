# Copyright (c) 2025, NVIDIA CORPORATION.
"""Run script for bulk-synchronous MPI shuffle."""

from __future__ import annotations

import argparse
from pathlib import Path

from rapidsmp.examples.bulk_shuffle import bulk_mpi_shuffle

parser = argparse.ArgumentParser(
    prog="Bulk-synchronous MPI shuffle",
    description="Shuffle a dataset at rest on both ends.",
)
parser.add_argument(
    "--input",
    type=str,
    default="/datasets/rzamora/data/sm_timeseries_pq",
    help="Input directory path.",
)
parser.add_argument(
    "--output",
    type=str,
    help="Output directory path.",
)
parser.add_argument(
    "--on",
    type=str,
    help="Comma-separated list of column names to shuffle on.",
)
parser.add_argument(
    "--n_output_files",
    type=int,
    default=None,
    help="Number of output files. Default preserves input file count.",
)
parser.add_argument(
    "--batchsize",
    type=int,
    default=1,
    help="Number of files to read on each MPI rank at once.",
)
parser.add_argument(
    "--baseline",
    default=False,
    action="store_true",
    help="Maximum device memory to use.",
)
args = parser.parse_args()


if __name__ == "__main__":
    bulk_mpi_shuffle(
        paths=sorted(map(str, Path(args.input).glob("**/*"))),
        shuffle_on=args.on.split(","),
        output_path=args.output,
        num_output_files=args.n_output_files,
        batchsize=args.batchsize,
        baseline=args.baseline,
    )
