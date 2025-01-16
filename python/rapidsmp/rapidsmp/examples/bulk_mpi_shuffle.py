# Copyright (c) 2025, NVIDIA CORPORATION.
"""Bulk-synchronous MPI shuffle."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING

from mpi4py import MPI

import pylibcudf as plc
import rmm.mr
from rmm._cuda.stream import DEFAULT_STREAM

from rapidsmp.buffer.resource import BufferResource
from rapidsmp.communicator.mpi import new_communicator
from rapidsmp.shuffler import Shuffler, partition_and_pack, unpack_and_concat
from rapidsmp.testing import to_cudf_dataframe

if TYPE_CHECKING:
    from collections.abc import Callable


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
args = parser.parse_args()


def read_batch(paths: list[str]) -> tuple[plc.Table, list[str]]:
    """Read a single batch of Parquet files."""
    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo(paths)
    ).build()
    tbl_w_meta = plc.io.parquet.read_parquet(options)
    return (tbl_w_meta.tbl, tbl_w_meta.column_names(include_children=False))


def write_table(table: plc.Table, output_path: str, id: int):
    """Write a pylibcudf Table to a Parquet file."""
    path = f"{output_path}/part.{id}.parquet"
    to_cudf_dataframe(table).to_parquet(path)


def bulk_mpi_shuffle(
    paths: list[str],
    shuffle_on: list[str],
    output_path: str,
    comm: MPI.Intercomm | None = None,
    num_output_files: int | None = None,
    batchsize: int | None = None,
    read_func: Callable | None = None,
    write_func: Callable | None = None,
) -> None:
    """
    Perform a bulk-synchronous dataset shuffle.

    Parameters
    ----------
    paths
        List of file paths to shuffle. This list contains all files in the
        dataset (not just the files that will be processed by the local rank).
    shuffle_on
        List of column names to shuffle on.
    output_path
        Path of the output directory where the data will be written. This
        directory does not need to be on a shared filesystem.
    comm
        Optional MPI communicator. Default is MPI.COMM_WORLD.
    num_output_files
        Number of output files to produce. Default will preserve the
        input file count.
    batchsize
        Number of files to read at once on each rank. Default is 1.
        TODO: Replace this argument with a blocksize mechanism that
        can also handle the splitting of large files.
    read_func
        Optional call-back function to read the input data. This function
        must accept a list of file paths, and return a pylibcudf Table and
        the list of column names in the table. Default logic will use
        `pylibcudf.read_parquet`.
    write_func
        Optional call-back function to write shuffled data to disk.
        must accept `table`, `output_path`, and `id` arguments. Default
        logic will write the pylibcudf table to a parquet file
        (e.g. `f"{output_path}/part.{id}.parquet"`).

    Notes
    -----
    This function is executed on each rank of the MPI communicator in a
    bulk-synchronous fashion. This means all ranks are expected to call
    this same function with the same arguments.
    """
    # Extract communicator and rank
    mpi_comm = comm or MPI.COMM_WORLD
    comm = new_communicator(mpi_comm)
    size = mpi_comm.size
    rank = comm.rank

    # Create output directory if necessary
    Path(output_path).mkdir(exist_ok=True)

    # Create buffer resource and shuffler
    br = BufferResource(rmm.mr.get_current_device_resource())
    num_input_files = len(paths)
    num_output_files = num_output_files or num_input_files
    total_num_partitions = num_output_files
    shuffler = Shuffler(
        comm,
        total_num_partitions=total_num_partitions,
        stream=DEFAULT_STREAM,
        br=br,
    )

    # Determine which files to process on this rank
    files_per_rank = math.ceil(num_input_files / size)
    start = files_per_rank * rank
    finish = start + files_per_rank
    local_files = paths[start:finish]
    num_local_files = len(local_files)
    batchsize = batchsize or 1
    num_batches = math.ceil(num_local_files / batchsize)

    # Read batches and submit them to the shuffler
    read_func = read_func or read_batch
    for batch_id in range(num_batches):
        batch = local_files[batch_id * batchsize : (batch_id + 1) * batchsize]
        table, columns = read_batch(batch)
        columns_to_hash = tuple(columns.index(val) for val in shuffle_on)
        packed_inputs = partition_and_pack(
            table,
            columns_to_hash=columns_to_hash,
            num_partitions=total_num_partitions,
        )
        shuffler.insert_chunks(packed_inputs)

    # Tell the shuffler we are done adding local data
    for pid in range(total_num_partitions):
        shuffler.insert_finished(pid)

    # Write shuffled partitions to disk as they finish
    write_func = write_func or write_table
    while not shuffler.finished():
        partition_id = shuffler.wait_any()
        table = unpack_and_concat(shuffler.extract(partition_id))
        write_func(
            table,
            output_path,
            partition_id,
        )
    shuffler.shutdown()


if __name__ == "__main__":
    bulk_mpi_shuffle(
        paths=sorted(map(str, Path(args.input).glob("**/*"))),
        shuffle_on=args.on.split(","),
        output_path=args.output,
        num_output_files=args.n_output_files,
    )
