# Copyright (c) 2025, NVIDIA CORPORATION.
"""Example performing a streaming shuffle."""

from __future__ import annotations

import argparse

import cupy as cp

import cudf
import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmp.buffer.resource import BufferResource
from rapidsmp.communicator.ucxx import new_communicator
from rapidsmp.shuffler import Shuffler, partition_and_pack, unpack_and_concat
from rapidsmp.statistics import Statistics


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
        "--n-partitions", type=int, default=10, help="Number of partitions"
    )
    parser.add_argument(
        "--n-rows", type=int, default=100, help="Number of rows per partition"
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


def main(args: list[str] | None = None) -> None:  # numpydoc ignore=PR01
    """Example performing a streaming shuffle."""
    parsed = parse_args(args)

    n_partitions = parsed.n_partitions
    n_rows = parsed.n_rows
    operation_id = 0

    comm = new_communicator(nranks=1, ucx_worker=None, root_ucxx_address=None)

    # Create a RMM stack with both a device pool and statistics.
    mr = rmm.mr.StatisticsResourceAdaptor(
        rmm.mr.PoolMemoryResource(
            rmm.mr.CudaMemoryResource(),
        )
    )
    rmm.mr.set_current_device_resource(mr)
    br = BufferResource(mr)
    statistics = Statistics(enable=True)
    stream = DEFAULT_STREAM
    shuffler = Shuffler(
        comm=comm,
        op_id=operation_id,
        total_num_partitions=n_partitions,
        stream=stream,
        br=br,
        statistics=statistics,
    )

    for partition_id in range(n_partitions):
        if parsed.progress:
            print(f"Inserting partition {partition_id}", end="\r")
        df = generate_partition(n_rows)
        (table, _) = df.to_pylibcudf()
        packed_inputs = partition_and_pack(
            table,
            columns_to_hash=[1],
            num_partitions=n_partitions,
            stream=stream,
            device_mr=rmm.mr.get_current_device_resource(),
        )
        shuffler.insert_chunks(packed_inputs)

    for partition_id in range(n_partitions):
        shuffler.insert_finished(partition_id)

    if parsed.progress:
        print("\nShuffling...", flush=True)

    while not shuffler.finished():
        partition_id = shuffler.wait_any()
        unpack_and_concat(
            shuffler.extract(partition_id),
            stream=DEFAULT_STREAM,
            device_mr=rmm.mr.get_current_device_resource(),
        )
        if parsed.progress:
            print(f"Finished partition {partition_id}", end="\r")

    if parsed.progress:
        print("\nDone!", flush=True)
    if parsed.report:
        print(statistics.report())


if __name__ == "__main__":
    main()
