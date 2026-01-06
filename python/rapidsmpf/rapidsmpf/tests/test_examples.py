# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

import cudf

from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.testing import assert_eq

MPI = pytest.importorskip("mpi4py.MPI")
from rapidsmpf.examples.bulk_mpi_shuffle import bulk_mpi_shuffle  # noqa: E402

if TYPE_CHECKING:
    import py.path

    import rmm.mr

    from rapidsmpf.communicator.communicator import Communicator


@pytest.mark.parametrize("batchsize", [1, 2, 3])
@pytest.mark.parametrize("num_output_files", [10, 5])
def test_bulk_shuffle(
    comm: Communicator,
    tmpdir: py.path.local.LocalPath,
    device_mr: rmm.mr.CudaMemoryResource,
    batchsize: int,
    num_output_files: int,
) -> None:
    # Get mpi-compatible tmpdir
    mpi_comm = MPI.COMM_WORLD
    rank = comm.rank
    name = str(tmpdir) if rank == 0 else None
    name = mpi_comm.bcast(name, root=0)
    mpi_tmpdir = type(tmpdir)(name)

    # Generate input dataset
    num_files = 15
    num_rows = 100
    np.random.seed(42)
    dataset_dir = mpi_tmpdir.join("dataset")
    if rank == 0:
        mpi_tmpdir.mkdir("dataset")
        for i in range(num_files):
            cudf.DataFrame(
                {
                    "a": range(i * num_rows, (i + 1) * num_rows),
                    "b": np.random.randint(0, 1000, num_rows),
                    "c": [i] * num_rows,
                }
            ).to_parquet(dataset_dir.join(f"part.{i}.parquet"))
        mpi_tmpdir.mkdir("output")
        input_paths = sorted(map(str, Path(dataset_dir).glob("**/*")))
    else:
        input_paths = None
    input_paths = mpi_comm.bcast(input_paths, root=0)
    assert isinstance(input_paths, list)  # for mypy
    output_dir = str(mpi_tmpdir.join("output"))

    # Use a default buffer resource.
    br = BufferResource(device_mr)

    # Perform a the shuffle
    bulk_mpi_shuffle(
        paths=input_paths,
        shuffle_on=["b"],
        output_path=output_dir,
        comm=comm,
        br=br,
        batchsize=batchsize,
        num_output_files=num_output_files,
    )
    mpi_comm.barrier()

    # Check that original and shuffled data match
    if rank == 0:
        shuffled_paths = sorted(map(str, Path(output_dir).glob("**/*")))
        df_original = cudf.read_parquet(input_paths)
        df_shuffled = cudf.read_parquet(shuffled_paths)
        assert_eq(df_original, df_shuffled, sort_rows="a")
    mpi_comm.barrier()
