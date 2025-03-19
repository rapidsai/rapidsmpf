# Copyright (c) 2025, NVIDIA CORPORATION.
"""Example performing a streaming shuffle."""

from __future__ import annotations

import argparse
import threading
from typing import TYPE_CHECKING

import cupy as cp
import ucxx._lib.libucxx as ucx_api
from mpi4py import MPI

import cudf
import rmm.mr
from pylibcudf.contiguous_split import pack
from rmm.pylibrmm.stream import DEFAULT_STREAM

import rapidsmp.communicator.mpi
from rapidsmp.buffer.buffer import MemoryType
from rapidsmp.buffer.resource import BufferResource, LimitAvailableMemory
from rapidsmp.communicator.ucxx import (
    barrier,
    get_root_ucxx_address,
    new_communicator,
)
from rapidsmp.shuffler import Shuffler
from rapidsmp.statistics import Statistics
from rapidsmp.utils.string import format_bytes, parse_bytes

if TYPE_CHECKING:
    from rapidsmp.communicator.communicator import Communicator


def generate_partition(size_bytes: int) -> cudf.DataFrame:
    """
    Generate a random partition of data.

    Parameters
    ----------
    size_bytes : int
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


def parse_args(
    args: list[str] | None = None,
) -> argparse.Namespace:  # numpydoc ignore=PR01,RT01
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--out-parts",
        type=int,
        help="Number of output partitions. Default: n_ranks in the cluster",
        default=None,
    )

    parser.add_argument(
        "--local-sz",
        type=parse_bytes,
        default="1MiB",
        help="Local data size. Default: 1MiB",
    )

    parser.add_argument(
        "--part-sz",
        type=parse_bytes,
        default=-1,
        help="Partition size. Local size will be split into partitions of this size. Default: local_sz.",
    )
    parser.add_argument(
        "--comm",
        type=str,
        default="mpi",
        help="Communicator type",
        choices={"mpi", "ucx"},
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
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the shuffling progress",
    )

    parser.add_argument(
        "--statistics",
        default=False,
        action="store_true",
        help="Enable statistics.",
    )
    return parser.parse_args(args)


def consume_finished_partitions(
    event: threading.Event,
    total_partitions: int,
    comm: Communicator,
    shuffler: Shuffler,
) -> None:
    """
    Consume the finished partitions.

    Parameters
    ----------
    event
        Threading event to signal the main thread that the consumer thread is done.
    total_partitions
        The total number of partitions.
    comm
        The communicator to use.
    shuffler
        The shuffler to use.
    """
    finished = 0
    while not shuffler.finished():
        partition_id = shuffler.wait_any()
        assert partition_id % comm.nranks() == comm.rank()

        splits = shuffler.extract(partition_id)
        del splits  # discard the extracted partition splits
        finished += 1

    assert finished == total_partitions / comm.nranks()

    event.set()  # notify finished


def streaming_shuffle(
    comm: Communicator,
    br: BufferResource,
    stats: Statistics,
    output_nparts: int,
    local_size: int,
    part_size: int,
) -> None:
    """
    Run shuffle opeartion in a streaming fashion.

    Main thread will produce local partitions and stream them through the shuffler. A separate
    consumer thread will consume the finished partitions, and discard them.

    Parameters
    ----------
    comm
        The communicator to use.
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
        op_id=0,
        total_num_partitions=output_nparts,
        stream=DEFAULT_STREAM,
        br=br,
        statistics=stats,
    )

    event = threading.Event()  # event to signal the consumer thread has finished

    # create a thread to consume the finished partitions
    consumer_thread = threading.Thread(
        target=consume_finished_partitions,
        args=(event, output_nparts, comm, shuffler),
    )
    consumer_thread.start()

    n_parts_local = local_size // part_size

    # simulate a hash partition by splitting a partition into total_num_partitions
    split_size = part_size // output_nparts
    dummy_table = generate_partition(split_size).to_pylibcudf()

    for _ in range(n_parts_local):
        # generate chunks for a single local partition by deep copying the dummy table as packed columns
        chunks = {}
        for i in range(output_nparts):
            chunks[i] = pack(dummy_table)

        shuffler.insert_chunks(chunks)

    # finish inserting all partitions
    for i in range(output_nparts):
        shuffler.insert_finished(i)

    # wait for all partitions to be consumed
    event.wait()

    consumer_thread.join()


def ucxx_mpi_setup() -> Communicator:
    """
    Bootstrap UCXX cluster using MPI.

    Returns
    -------
    Communicator
        A new ucxx communicator.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        comm = new_communicator(MPI.COMM_WORLD.size, None, None)
        root_address_str = get_root_ucxx_address(comm)
    else:
        root_address_str = None

    root_address_str = MPI.COMM_WORLD.bcast(root_address_str, root=0)

    if MPI.COMM_WORLD.Get_rank() != 0:
        root_address = ucx_api.UCXAddress.create_from_buffer(root_address_str)
        comm = new_communicator(MPI.COMM_WORLD.size, None, root_address)

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
    if args.comm == "mpi":
        comm = rapidsmp.communicator.mpi.new_communicator(MPI.COMM_WORLD)
    elif args.comm == "ucxx":
        comm = ucxx_mpi_setup()

    # Create a RMM stack with both a device pool and statistics.
    mr = rmm.mr.StatisticsResourceAdaptor(
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

    stats = Statistics(args.statistics)

    out_nparts: int = args.out_parts if args.out_parts is not None else comm.nranks
    local_size: int = args.local_sz
    part_size: int = args.part_sz if args.part_sz > 0 else local_size

    MPI.COMM_WORLD.barrier()
    start_time = MPI.Wtime()
    streaming_shuffle(comm, br, stats, out_nparts, local_size, part_size)
    elapsed_time = MPI.Wtime() - start_time
    MPI.COMM_WORLD.barrier()

    mem_peak = format_bytes(mr.allocation_counts.peak_bytes)
    comm.logger.print(
        f"elapsed: {elapsed_time:.2f} sec | rmm device memory peak: {mem_peak}"
    )


def main(args: list[str] | None = None) -> None:  # numpydoc ignore=PR01
    """Example performing a streaming shuffle."""
    print(
        parse_bytes(
            format_bytes((int(rmm.mr.available_device_memory()[1] * 0.8) // 256) * 256)
        )
    )

    parsed = parse_args(args)

    print(parsed)

    setup_and_run(parsed)


if __name__ == "__main__":
    main()
