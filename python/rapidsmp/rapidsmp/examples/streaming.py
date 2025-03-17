# Copyright (c) 2025, NVIDIA CORPORATION.
"""Example performing a streaming shuffle."""

from __future__ import annotations

import argparse

import cupy as cp

from mpi4py import MPI

import cudf
import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

import rapidsmp.communicator.mpi
from rapidsmp.communicator import Communicator
from rapidsmp.buffer.resource import BufferResource
from rapidsmp.buffer.buffer import MemoryType
from rapidsmp.buffer.resource import BufferResource, LimitAvailableMemory
from rapidsmp.communicator.ucxx import new_communicator
from rapidsmp.shuffler import Shuffler, partition_and_pack, unpack_and_concat
from rapidsmp.statistics import Statistics
from rapidsmp.utils.string import parse_bytes, format_bytes


def generate_partition(n_rows: int = 100) -> cudf.DataFrame:
    """
    Generate a random partition of data.

    Parameters
    ----------
    n_rows : int
        The length of the dataframe.

    Returns
    -------
    cudf.DataFrame
    """
    return cudf.DataFrame(
        {
            "id": cp.arange(n_rows),
            "group": cp.random.randint(0, 10, size=n_rows),
            "value": cp.random.uniform(size=n_rows),
        }
    )


def parse_args(
    args: list[str] | None = None,
) -> argparse.Namespace:  # numpydoc ignore=PR01,RT01
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--out-parts",
        type=int | None,
        help="Number of output partitions. Default: n_ranks in the cluster",
        default=None,
    )

    parser.add_argument(
        "--local-sz", type=parse_bytes, default="1GiB", help="Local data size"
    )

    parser.add_argument(
        "--part-sz",
        type=parse_bytes,
        default="1GiB",
        help="Partition size. Local size will be split into partitions of this size",
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
    return parser.parse_args(args)


def ucxx_mpi_setup() -> Communicator:
    """
    Bootstrap UCXX cluster using MPI.

    Returns
    -------
    Communicator
        A new ucxx communicator.
    """
    import ucxx._lib.libucxx as ucx_api

    from rapidsmp.communicator.ucxx import (
        barrier,
        get_root_ucxx_address,
        new_communicator,
    )

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
    """
    if args.cluster_type == "mpi":
        comm = rapidsmp.communicator.mpi.new_communicator(MPI.COMM_WORLD)
    elif args.cluster_type == "ucxx":
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

    MPI.COMM_WORLD.barrier()
    start_time = MPI.Wtime()
    streaming_shuffle(
        comm=comm,
        br=br,
        statistics=stats,
    )
    elapsed_time = MPI.Wtime() - start_time
    MPI.COMM_WORLD.barrier()



def main(args: list[str] | None = None) -> None:  # numpydoc ignore=PR01
    """Example performing a streaming shuffle."""
    parsed = parse_args(args)

    setup_and_run(parsed)

if __name__ == "__main__":
    main()
