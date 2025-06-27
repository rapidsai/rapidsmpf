# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
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

import rapidsmpf.communicator.mpi
from rapidsmpf.buffer.buffer import MemoryType
from rapidsmpf.buffer.resource import BufferResource, LimitAvailableMemory
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.integrations.cudf.partition import (
    partition_and_pack,
    unpack_and_concat,
)
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.shuffler import Shuffler
from rapidsmpf.statistics import Statistics
from rapidsmpf.testing import pylibcudf_to_cudf_dataframe
from rapidsmpf.utils.string import format_bytes, parse_bytes

if TYPE_CHECKING:
    from collections.abc import Callable

    from rapidsmpf.communicator.communicator import Communicator


def read_batch(paths: list[str]) -> tuple[plc.Table, list[str]]:
    """
    Read a single batch of Parquet files.

    Parameters
    ----------
    paths
        List of file paths to the Parquet files.

    Returns
    -------
    plc.Table
        The table containing the data read from the Parquet files.
    list of str
        Column names from the Parquet files, excluding nested children.
    """
    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo(paths)
    ).build()
    tbl_w_meta = plc.io.parquet.read_parquet(options)
    return (tbl_w_meta.tbl, tbl_w_meta.column_names(include_children=False))


def write_table(
    table: plc.Table, output_path: str, id: int | str, column_names: list[str] | None
) -> None:
    """
    Write a pylibcudf Table to a Parquet file.

    Parameters
    ----------
    table
        The table to be written to the Parquet file.
    output_path : str
        Directory where the Parquet file will be written.
    id
        Unique identifier used to generate the filename using `part.{id}.parque`.
    column_names
        List of column names.
    """
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
    statistics: Statistics | None = None,
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
    statistics
        The statistics instance to use. If None, statistics is disabled.

    Notes
    -----
    This function is executed on each rank of the MPI communicator in a
    bulk-synchronous fashion. This means all ranks are expected to call
    this same function with the same arguments.
    """
    # Create output directory if necessary
    Path(output_path).mkdir(exist_ok=True)

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
        progress_thread = ProgressThread(comm)

        shuffler = Shuffler(
            comm,
            progress_thread,
            op_id=0,
            total_num_partitions=total_num_partitions,
            stream=DEFAULT_STREAM,
            br=br,
            statistics=statistics,
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


def ucxx_mpi_setup(options: Options) -> Communicator:
    """
    Bootstrap UCXX cluster using MPI.

    Returns
    -------
    Communicator
        A new ucxx communicator.
    options
        Configuration options.
    """
    import ucxx._lib.libucxx as ucx_api

    from rapidsmpf.communicator.ucxx import (
        barrier,
        get_root_ucxx_address,
        new_communicator,
    )

    if MPI.COMM_WORLD.Get_rank() == 0:
        comm = new_communicator(MPI.COMM_WORLD.size, None, None, options)
        root_address_bytes = get_root_ucxx_address(comm)
    else:
        root_address_bytes = None

    root_address_bytes = MPI.COMM_WORLD.bcast(root_address_bytes, root=0)

    if MPI.COMM_WORLD.Get_rank() != 0:
        root_address = ucx_api.UCXAddress.create_from_buffer(root_address_bytes)
        comm = new_communicator(MPI.COMM_WORLD.size, None, root_address, options)

    assert comm.nranks == MPI.COMM_WORLD.size
    barrier(comm)
    return comm


def setup_and_run(args: argparse.Namespace) -> None:
    """
    Set up the environment and run the shuffle example.

    Parameters
    ----------
    args
        Command-line arguments containing the configuration for the shuffle example.
    """
    options = Options(get_environment_variables())

    if args.cluster_type == "mpi":
        comm = rapidsmpf.communicator.mpi.new_communicator(MPI.COMM_WORLD, options)
    elif args.cluster_type == "ucxx":
        comm = ucxx_mpi_setup(options)

    # Create a RMM stack with both a device pool and statistics.
    mr = RmmResourceAdaptor(
        rmm.mr.PoolMemoryResource(
            rmm.mr.CudaMemoryResource(),
            initial_pool_size=args.rmm_pool_size,
            maximum_pool_size=args.rmm_pool_size,
        )
    )
    rmm.mr.set_current_device_resource(mr)

    # Create a buffer resource that limits device memory if `--spill-device`
    # is not None.
    memory_available = (
        None
        if args.spill_device is None
        else {MemoryType.DEVICE: LimitAvailableMemory(mr, limit=args.spill_device)}
    )
    br = BufferResource(mr, memory_available)

    stats = Statistics(enable=args.statistics, mr=mr)

    if comm.rank == 0:
        spill_device = (
            "disabled" if args.spill_device is None else format_bytes(args.spill_device)
        )
        comm.logger.print(
            f"""\
Shuffle:
    input: {args.input}
    output: {args.output}
    on: {args.on}
  --cluster-type: {args.cluster_type}
  --n-output-files: {args.n_output_files}
  --batchsize: {args.batchsize}
  --baseline: {args.baseline}
  --rmm-pool-size: {format_bytes(args.rmm_pool_size)}
  --spill-device: {spill_device}"""
        )

    MPI.COMM_WORLD.barrier()
    start_time = MPI.Wtime()
    bulk_mpi_shuffle(
        paths=sorted(map(str, args.input.glob("**/*"))),
        shuffle_on=args.on.split(","),
        output_path=args.output,
        comm=comm,
        br=br,
        num_output_files=args.n_output_files,
        batchsize=args.batchsize,
        baseline=args.baseline,
        statistics=stats,
    )
    elapsed_time = MPI.Wtime() - start_time
    MPI.COMM_WORLD.barrier()

    mem_peak = format_bytes(mr.get_main_record().peak())
    comm.logger.print(
        f"elapsed: {elapsed_time:.2f} sec | rmm device memory peak: {mem_peak}"
    )
    if stats.enabled:
        comm.logger.print(stats.report())


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
        metavar="INPUT_DIR_PATH",
        help="Input directory path.",
    )
    parser.add_argument(
        "output",
        metavar="OUTPUT_DIR_PATH",
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
        "--n-output-files",
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
        help="Run an IO baseline without any shuffling.",
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
    parser.add_argument(
        "--statistics",
        default=False,
        action="store_true",
        help="Enable statistics.",
    )
    parser.add_argument(
        "--cluster-type",
        type=str,
        default="mpi",
        choices=("mpi", "ucxx"),
        help=(
            "Cluster type to setup. Regardless of the cluster type selected it must "
            "be launched with 'mpirun'."
        ),
    )
    args = parser.parse_args()
    args.rmm_pool_size = (args.rmm_pool_size // 256) * 256  # Align to 256 bytes
    setup_and_run(args)
