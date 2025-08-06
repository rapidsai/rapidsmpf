# Copyright (c) 2025, NVIDIA CORPORATION.
"""Example performing a streaming shuffle."""

from __future__ import annotations

import argparse
import threading
import time
from typing import TYPE_CHECKING

import cupy as cp
import ucxx._lib.libucxx as ucx_api
from mpi4py import MPI

import cudf
import rmm.mr
from pylibcudf.contiguous_split import pack
from rmm.pylibrmm.stream import DEFAULT_STREAM

import rapidsmpf.communicator.mpi
from rapidsmpf.buffer.buffer import MemoryType
from rapidsmpf.buffer.packed_data import PackedData
from rapidsmpf.buffer.resource import BufferResource, LimitAvailableMemory
from rapidsmpf.communicator.ucxx import (
    barrier,
    get_root_ucxx_address,
    new_communicator,
)
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.shuffler import Shuffler
from rapidsmpf.statistics import Statistics
from rapidsmpf.utils.string import format_bytes, parse_bytes

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator


def generate_partition(size_bytes: int) -> cudf.DataFrame:
    """
    Generate a random partition of data.

    Parameters
    ----------
    size_bytes
        size of the dataframe in bytes

    Returns
    -------
    cudf.DataFrame
    """
    n_rows = size_bytes // 8  # each row is 8 bytes
    return cudf.DataFrame(
        {
            "id": cp.arange(0, n_rows, dtype=cp.int32),
            "value": cp.arange(0, n_rows, dtype=cp.float32),
        }
    )


def consume_finished_partitions(
    total_partitions: int,
    comm: Communicator,
    shuffler: Shuffler,
) -> None:
    """
    Consume the finished partitions.

    Parameters
    ----------
    total_partitions
        The total number of partitions.
    comm
        The communicator to use.
    shuffler
        The shuffler to use.
    """
    finished = set()
    while not shuffler.finished():
        partition_id = shuffler.wait_any()
        assert partition_id % comm.nranks == comm.rank

        # discard the extracted partition splits
        shuffler.extract(partition_id)

        finished.add(partition_id)

    # all gather len(finished) to determine if all partitions have finished
    comm.logger.print(f"Received parts: {len(finished)}")
    finished_parts: int = MPI.COMM_WORLD.allreduce(len(finished), op=MPI.SUM)
    assert finished_parts == total_partitions, "all partitions have not finished"


def streaming_shuffle(
    comm: Communicator,
    progress_thread: ProgressThread,
    br: BufferResource,
    stats: Statistics,
    output_nparts: int,
    local_size: int,
    part_size: int,
    insert_delay_ms: float,
    wait_timeout: int | None,
) -> None:
    """
    Run shuffle operation in a streaming fashion.

    Main thread will produce local partitions and stream them through the shuffler. A separate
    consumer thread will consume the finished partitions, and discard them.

    Parameters
    ----------
    comm
        The communicator to use.
    progress_thread
        The progress thread to use.
    br
        The buffer resource to use.
    stats
        The statistics to use.
    output_nparts
        The total number of output partitions.
    local_size
        The size of the local partition.
    part_size
        The size of each partition.
    insert_delay_ms
        A delay (ms) before inserting a partition to the shuffler.
    wait_timeout
        Timeout to wait for completion
    """
    assert local_size >= part_size, "local_size must be >= part_size"
    assert local_size % part_size == 0, "local_size must be divisible by part_size"
    assert part_size >= 8 * output_nparts, "part_size must be >= 8 * output_nparts"
    assert part_size % output_nparts == 0, (
        "part_size must be divisible by output_nparts"
    )

    # create a shuffler instance
    shuffler = Shuffler(
        comm,
        progress_thread,
        op_id=0,
        total_num_partitions=output_nparts,
        stream=DEFAULT_STREAM,
        br=br,
        statistics=stats,
    )

    # create a thread to consume the finished partitions. It is a daemon thread, so it
    # will not block the main thread from exiting in case of an error.
    consumer_thread = threading.Thread(
        target=consume_finished_partitions,
        args=(output_nparts, comm, shuffler),
        daemon=True,
    )

    # start the consumer thread. This will wait for any finished partition.
    consumer_thread.start()

    n_parts_local = local_size // part_size

    # simulate a hash partition by splitting a partition into total_num_partitions
    split_size = part_size // output_nparts
    dummy_table, _ = generate_partition(split_size).to_pylibcudf()

    comm.logger.print(f"num local partitions:{n_parts_local} split size:{split_size}")
    for p in range(n_parts_local):
        # generate chunks for a single local partition by deep copying the dummy table
        # as packed columns
        # NOTE: This will require part_size amount of GPU memory.
        chunks: dict[int, PackedData] = {}
        for i in range(output_nparts):
            chunks[i] = PackedData.from_cudf_packed_columns(pack(dummy_table))

        if p > 0 and insert_delay_ms > 0:
            time.sleep(insert_delay_ms / 1000)

        shuffler.insert_chunks(chunks)
    # finish inserting all partitions
    for i in range(output_nparts):
        shuffler.insert_finished(i)

    # wait for the consumer thread to finish.
    consumer_thread.join(timeout=wait_timeout)


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
    if MPI.COMM_WORLD.Get_rank() == 0:
        comm = new_communicator(MPI.COMM_WORLD.size, None, None, options)
        root_address_str = get_root_ucxx_address(comm)
    else:
        root_address_str = None

    root_address_str = MPI.COMM_WORLD.bcast(root_address_str, root=0)

    if MPI.COMM_WORLD.Get_rank() != 0:
        root_address = ucx_api.UCXAddress.create_from_buffer(root_address_str)
        comm = new_communicator(MPI.COMM_WORLD.size, None, root_address, options)

    assert comm.nranks == MPI.COMM_WORLD.size
    barrier(comm)
    return comm


def setup_and_run(args: argparse.Namespace) -> None:
    """
    Setup the args.

    Parameters
    ----------
    args
        The arguments to parse.
    """
    options = Options(get_environment_variables())

    if args.comm == "mpi":
        comm = rapidsmpf.communicator.mpi.new_communicator(MPI.COMM_WORLD, options)
    elif args.comm == "ucxx":
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

    stats = Statistics(enable=args.statistics, mr=mr)
    progress_thread = ProgressThread(comm, stats)

    # Create a buffer resource that limits device memory if `--spill-device`
    # is not None.
    memory_available = (
        None
        if args.spill_device is None
        else {MemoryType.DEVICE: LimitAvailableMemory(mr, limit=args.spill_device)}
    )
    br = BufferResource(mr, memory_available)

    args.out_nparts = args.out_nparts if args.out_nparts is not None else comm.nranks
    args.part_size = args.part_size if args.part_size is not None else args.local_size

    if comm.rank == 0:
        comm.logger.print(str(vars(args)))

    MPI.COMM_WORLD.barrier()
    start_time = MPI.Wtime()
    streaming_shuffle(
        comm,
        progress_thread,
        br,
        stats,
        args.out_nparts,
        args.local_size,
        args.part_size,
        args.insert_delay_ms,
        args.wait_timeout,
    )
    elapsed_time = MPI.Wtime() - start_time
    MPI.COMM_WORLD.barrier()

    mem_peak = format_bytes(mr.get_main_record().peak())
    comm.logger.print(
        f"elapsed: {elapsed_time:.2f} sec | rmm device memory peak: {mem_peak}"
    )

    if args.statistics:
        comm.logger.print(stats.report())


def parse_args(
    args: list[str] | None = None,
) -> argparse.Namespace:  # numpydoc ignore=PR01,RT01
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="streaming shuffle", description="Streaming shuffle example"
    )

    parser.add_argument(
        "--out-nparts",
        type=int,
        help="Number of output partitions. Default: n_ranks in the cluster",
        default=None,
    )

    parser.add_argument(
        "--local-size",
        type=parse_bytes,
        default="1MiB",
        help="Local data size. Default: 1MiB",
    )

    parser.add_argument(
        "--part-size",
        type=parse_bytes,
        default=None,
        help="Partition size. Local size will be split into partitions of this size. Default: local_sz.",
    )
    parser.add_argument(
        "--comm",
        type=str,
        default="mpi",
        help="Communicator type",
        choices={"mpi", "ucxx"},
    )

    parser.add_argument(
        "--rmm-pool-size",
        type=parse_bytes,
        default=(int(parse_bytes(rmm.mr.available_device_memory()[1]) * 0.8) // 256)
        * 256,
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
        "--report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the statistics report",
    )

    parser.add_argument(
        "--statistics",
        default=False,
        action="store_true",
        help="Enable statistics.",
    )

    parser.add_argument(
        "--insert-delay-ms",
        type=float,
        help="A delay (ms) before inserting a partition to the shuffler. Default: 0",
        default=0,
    )

    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=None,
        help="Wait timeout to finish. Default, wait indefinitely",
    )

    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:  # numpydoc ignore=PR01
    """Streaming shuffle."""
    parsed = parse_args(args)
    setup_and_run(parsed)


if __name__ == "__main__":
    main()
