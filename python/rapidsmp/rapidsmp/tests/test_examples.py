# Copyright (c) 2025, NVIDIA CORPORATION.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import cudf

from rapidsmp.testing import assert_eq


@pytest.mark.parametrize("batchsize", [1, 2, 3])
@pytest.mark.parametrize("num_output_files", [10, 5])
def test_bulk_mpi_shuffle(comm, tmpdir, batchsize, num_output_files):
    from rapidsmp.examples.bulk_shuffle import bulk_mpi_shuffle

    # Generate input dataset
    num_files = 10
    num_rows = 100
    rank = comm.rank
    np.random.seed(42)
    dataset_dir = tmpdir.join("dataset")
    if rank == 0:
        tmpdir.mkdir("dataset")
        for i in range(num_files):
            cudf.DataFrame(
                {
                    "a": range(i * num_rows, (i + 1) * num_rows),
                    "b": np.random.randint(0, 1000, num_rows),
                    "c": [i] * num_rows,
                }
            ).to_parquet(dataset_dir.join(f"part.{i}.parquet"))
    input_paths = sorted(map(str, Path(dataset_dir).glob("**/*")))
    output_dir = str(tmpdir.join("output"))

    # Perform a the shuffle
    bulk_mpi_shuffle(
        paths=input_paths,
        shuffle_on=["b"],
        output_path=output_dir,
        batchsize=batchsize,
        num_output_files=num_output_files,
    )
    shuffled_paths = sorted(map(str, Path(output_dir).glob("**/*")))

    # Check that original and shuffled data match
    df_original = cudf.read_parquet(input_paths)
    df_shuffled = cudf.read_parquet(shuffled_paths)
    assert_eq(df_original, df_shuffled, sort_rows="a")
