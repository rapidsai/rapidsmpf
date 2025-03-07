"""Example performing a streaming shuffle."""

from __future__ import annotations

import cupy as cp

import cudf
import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmp.buffer.resource import BufferResource
from rapidsmp.communicator.ucxx import new_communicator
from rapidsmp.shuffler import Shuffler, partition_and_pack, unpack_and_concat
from rapidsmp.statistics import Statistics


def generate_partition(n_rows: int = 100) -> cudf.DataFrame:
    """Generate a random partition of data."""
    return cudf.DataFrame(
        {
            "id": cp.arange(n_rows),
            "group": cp.random.randint(0, 10, size=n_rows),
            "value": cp.random.uniform(size=n_rows),
        }
    )


def main():
    n_partitions = 10
    n_rows = 100
    operation_id = 0

    comm = new_communicator(1, None, None)

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
        print(f"Inserting partition {partition_id}")
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

    while not shuffler.finished():
        partition_id = shuffler.wait_any()
        unpack_and_concat(
            shuffler.extract(partition_id),
            stream=DEFAULT_STREAM,
            device_mr=rmm.mr.get_current_device_resource(),
        )
        print(f"Finished partition {partition_id}")


if __name__ == "__main__":
    main()
