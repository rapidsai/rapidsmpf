# Copyright (c) 2025, NVIDIA CORPORATION.
"""Bulk-synchronous MPI shuffle."""

from __future__ import annotations

import argparse
import math
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from mpi4py import MPI

import pylibcudf as plc
import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

import rapidsmp.communicator.mpi
from rapidsmp.buffer.buffer import MemoryType
from rapidsmp.buffer.resource import BufferResource, LimitAvailableMemory
from rapidsmp.shuffler import Shuffler, partition_and_pack, unpack_and_concat
from rapidsmp.testing import pylibcudf_to_cudf_dataframe
from rapidsmp.utils.string import format_bytes, parse_bytes

if TYPE_CHECKING:
    from collections.abc import Callable

    from rapidsmp.communicator.communicator import Communicator


def read_batch(paths: list[str]) -> tuple[plc.Table, list[str]]:
    """Read a single batch of Parquet files."""
    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo(paths)
    ).build()
    tbl_w_meta = plc.io.parquet.read_parquet(options)
    return (tbl_w_meta.tbl, tbl_w_meta.column_names(include_children=False))


def write_table(
    table: plc.Table, output_path: str, id: int | str, column_names: list[str] | None
):
    """Write a pylibcudf Table to a Parquet file."""
    path = f"{output_path}/part.{id}.parquet"
    pylibcudf_to_cudf_dataframe(
        table,
        column_names=column_names,
    ).to_parquet(path)


def bulk_mpi_shuffle(
    paths: list[str],
    shuffle_on: list[str],
    output_path: str,
    comm: Communicator,
    br: BufferResource,
    *,
    num_output_files: int | None = None,
    batchsize: int = 1,
    read_func: Callable = read_batch,
    write_func: Callable = write_table,
    baseline: bool = False,
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
        The communicator to use.
    br
        Buffer resource to use.
    num_output_files
        Number of output files to produce. Default will preserve the
        input file count.
    batchsize
        Number of files to read at once on each rank.
    read_func
        Call-back function to read the input data. This function must accept a
        list of file paths, and return a pylibcudf Table and the list of column
        names in the table. Default logic will use `pylibcudf.read_parquet`.
    write_func
        Call-back function to write shuffled data to disk. This function must
        accept `table`, `output_path`, `id`, and `column_names` arguments.
        Default logic will write the pylibcudf table to a parquet file
        (e.g. `f"{output_path}/part.{id}.parquet"`).
    baseline
        Whether to skip the shuffle and run a simple IO baseline.

    Notes
    -----
    This function is executed on each rank of the MPI communicator in a
    bulk-synchronous fashion. This means all ranks are expected to call
    this same function with the same arguments.
    """
    # Create output directory if necessary
    Path(output_path).mkdir(exist_ok=True)

    if comm.rank == 0:
        start_time = MPI.Wtime()

    # Determine which files to process on this rank
    num_input_files = len(paths)
    num_output_files = num_output_files or num_input_files
    total_num_partitions = num_output_files
    files_per_rank = math.ceil(num_input_files / comm.nranks)
    start = files_per_rank * comm.rank
    finish = start + files_per_rank
    local_files = paths[start:finish]
    num_local_files = len(local_files)
    num_batches = math.ceil(num_local_files / batchsize)

    if baseline:
        # Skip the shuffle - Run IO baseline
        for batch_id in range(num_batches):
            batch = local_files[batch_id * batchsize : (batch_id + 1) * batchsize]
            table, columns = read_func(batch)
            write_func(
                table,
                output_path,
                str(uuid.uuid4()),
                columns,
            )
    else:
        shuffler = Shuffler(
            comm,
            op_id=0,
            total_num_partitions=total_num_partitions,
            stream=DEFAULT_STREAM,
            br=br,
        )

        # Read batches and submit them to the shuffler
        column_names = None
        for batch_id in range(num_batches):
            batch = local_files[batch_id * batchsize : (batch_id + 1) * batchsize]
            table, columns = read_func(batch)
            if column_names is None:
                column_names = columns
            columns_to_hash = tuple(columns.index(val) for val in shuffle_on)
            packed_inputs = partition_and_pack(
                table,
                columns_to_hash=columns_to_hash,
                num_partitions=total_num_partitions,
                stream=DEFAULT_STREAM,
                device_mr=rmm.mr.get_current_device_resource(),
            )
            shuffler.insert_chunks(packed_inputs)

        # Tell the shuffler we are done adding local data
        for pid in range(total_num_partitions):
            shuffler.insert_finished(pid)

        # Write shuffled partitions to disk as they finish
        while not shuffler.finished():
            partition_id = shuffler.wait_any()
            table = unpack_and_concat(
                shuffler.extract(partition_id),
                stream=DEFAULT_STREAM,
                device_mr=rmm.mr.get_current_device_resource(),
            )
            write_func(
                table,
                output_path,
                partition_id,
                column_names,
            )
        shuffler.shutdown()

    if comm.rank == 0:
        end_time = MPI.Wtime()
        print(f"Shuffle took {end_time - start_time} seconds")


def setup_and_run(args) -> None:
    """Setup the environment and run the shuffle example."""
    comm = rapidsmp.communicator.mpi.new_communicator(MPI.COMM_WORLD)

    # Create a RMM stack with both a device pool and statistics.
    mr = rmm.mr.StatisticsResourceAdaptor(
        rmm.mr.PoolMemoryResource(
            rmm.mr.CudaMemoryResource(),
            initial_pool_size=args.rmm_pool_size,
            maximum_pool_size=args.rmm_pool_size,
        )
    )

    # Create a buffer resource that limits device memory if `--spill-device`
    # is not None.
    memory_available = (
        None
        if args.spill_device is None
        else {MemoryType.DEVICE: LimitAvailableMemory(mr, limit=args.spill_device)}
    )
    br = BufferResource(mr, memory_available)

    MPI.COMM_WORLD.barrier()
    bulk_mpi_shuffle(
        paths=sorted(map(str, args.input.glob("**/*"))),
        shuffle_on=args.on.split(","),
        output_path=args.output,
        comm=comm,
        br=br,
        num_output_files=args.n_output_files,
        batchsize=args.batchsize,
        baseline=args.baseline,
    )
    MPI.COMM_WORLD.barrier()


def dir_path(path: str) -> Path:
    """
    Validate that the given path is a directory and return a Path object.

    Parameters
    ----------
    path
        The path to check.

    Returns
    -------
    Path
        A Path object representing the directory.
    """
    ret = Path(path)
    if not ret.is_dir():
        raise ValueError()
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Bulk-synchronous MPI shuffle",
        description="Shuffle a dataset at rest (on disk) on both ends.",
    )
    parser.add_argument(
        "input",
        type=dir_path,
        metavar="DIR_PATH",
        help="Input directory path.",
    )
    parser.add_argument(
        "output",
        metavar="DIR_PATH",
        type=Path,
        help="Output directory path.",
    )
    parser.add_argument(
        "on",
        metavar="COLUMN_LIST",
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
    parser.add_argument(
        "--rmm-pool-size",
        type=parse_bytes,
        default=format_bytes(int(rmm.mr.available_device_memory()[1] * 0.8)),
        help=(
            "The size of the RMM pool as a string with unit such as '2MiB' and '4KiB'. "
            "Default to 80%% of the total device memory, which is %(default)s."
        ),
    )
    parser.add_argument(
        "--spill-device",
        type=lambda x: None if x is None else parse_bytes(x),
        default=None,
        help=(
            "Spilling device-to-host threshold as a string with unit such as '2MiB' "
            "and '4KiB'. Default is no spilling"
        ),
    )
    args = parser.parse_args()
    args.rmm_pool_size = (args.rmm_pool_size // 256) * 256  # Align to 256 bytes
    setup_and_run(args)
