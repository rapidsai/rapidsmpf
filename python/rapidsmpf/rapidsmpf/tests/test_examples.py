# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

import pylibcudf as plc

from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.testing import assert_eq_with_pyarrow

MPI = pytest.importorskip("mpi4py.MPI")
from rapidsmpf.examples.bulk_mpi_shuffle import bulk_mpi_shuffle  # noqa: E402

if TYPE_CHECKING:
    import py.path

    import rmm.mr

    from rapidsmpf.communicator.communicator import Communicator


def _write_parquet(table: plc.Table, column_names: list[str], path: str) -> None:
    metadata = plc.io.types.TableInputMetadata(table)
    for col_meta, name in zip(metadata.column_metadata, column_names, strict=True):
        col_meta.set_name(name)
    options = (
        plc.io.parquet.ParquetWriterOptions.builder(plc.io.SinkInfo([path]), table)
        .metadata(metadata)
        .build()
    )
    plc.io.parquet.write_parquet(options)


def _read_parquet(paths: list[str]) -> plc.Table:
    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo(paths)
    ).build()
    return plc.io.parquet.read_parquet(options).tbl


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
            table = plc.Table(
                [
                    plc.Column.from_iterable_of_py(
                        list(range(i * num_rows, (i + 1) * num_rows)),
                        plc.DataType(plc.TypeId.INT64),
                    ),
                    plc.Column.from_array(np.random.randint(0, 1000, num_rows)),
                    plc.Column.from_iterable_of_py(
                        [i] * num_rows, plc.DataType(plc.TypeId.INT64)
                    ),
                ]
            )
            _write_parquet(
                table,
                ["a", "b", "c"],
                str(dataset_dir.join(f"part.{i}.parquet")),
            )
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
        df_original = _read_parquet(input_paths)
        df_shuffled = _read_parquet(shuffled_paths)
        assert_eq_with_pyarrow(df_original, df_shuffled, sort_rows=0)
    mpi_comm.barrier()
