# Copyright (c) 2025, NVIDIA CORPORATION.
"""Example running a RapidsMP Shuffle operation using Ray and UCXX communication."""

from __future__ import annotations

import argparse
import math

import numpy as np
import ray

import cudf
import rmm

from rapidsmp.buffer.resource import BufferResource
from rapidsmp.integrations.ray import (
    RapidsMPActor,
    setup_ray_ucxx_cluster,
)
from rapidsmp.shuffler import Shuffler, partition_and_pack, unpack_and_concat
from rapidsmp.testing import assert_eq
from rapidsmp.utils.cudf import (
    cudf_to_pylibcudf_table,
    pylibcudf_to_cudf_dataframe,
)


@ray.remote(num_gpus=1)
class ShufflingActor(RapidsMPActor):
    """Ray actor that performs a shuffle operation."""

    def __init__(self, nranks):
        super().__init__(nranks)

    def _setup_device_mr(self):
        mr = rmm.mr.CudaMemoryResource()
        rmm.mr.set_current_device_resource(mr)
        return mr

    def _gen_cudf(self, args):
        # Every rank creates the full input dataframe and all the expected partitions
        # (also partitions this rank might not get after the shuffle).

        np.random.seed(42)  # Make sure all ranks create the same input dataframe.

        return cudf.DataFrame(
            {
                "a": range(args.num_rows),
                "b": np.random.randint(0, 1000, args.num_rows),
                "c": ["cat", "dog"] * (args.num_rows // 2),
            }
        )

    def run(self, args) -> None:
        """Runs the shuffle operation, and this will be called remotely from the client."""
        if self.comm is None:
            raise RuntimeError("RapidsMP not initialized")

        # If DEFAULT_STREAM was imported outside of this context, it will be pickled,
        # and it is not serializable. Therefore, we need to import it here.
        from rmm.pylibrmm.stream import DEFAULT_STREAM

        # if args.total_nparts is not set, set it to the number of ranks
        if args.total_nparts == -1:
            args.total_nparts = self._nranks

        mr = self._setup_device_mr()
        br = BufferResource(mr)

        df = self._gen_cudf(args)

        columns_to_hash = (df.columns.get_loc("b"),)
        column_names = list(df.columns)

        # Calculate the expected output partitions on all ranks
        expected = {
            partition_id: pylibcudf_to_cudf_dataframe(
                unpack_and_concat(
                    [packed],
                    stream=DEFAULT_STREAM,
                    device_mr=mr,
                ),
                column_names=column_names,
            )
            for partition_id, packed in partition_and_pack(
                cudf_to_pylibcudf_table(df),
                columns_to_hash=columns_to_hash,
                num_partitions=args.total_nparts,
                stream=DEFAULT_STREAM,
                device_mr=mr,
            ).items()
        }

        # Create shuffler
        shuffler = Shuffler(
            self.comm,
            op_id=0,
            total_num_partitions=args.total_nparts,
            stream=DEFAULT_STREAM,
            br=br,
        )

        # Slice df and submit local slices to shuffler
        stride = math.ceil(args.num_rows / self.comm.nranks)
        local_df = df.iloc[self.comm.rank * stride : (self.comm.rank + 1) * stride]
        num_rows_local = len(local_df)
        args.batch_sz = args.batch_sz if args.batch_sz > 0 else num_rows_local
        for i in range(0, num_rows_local, args.batch_sz):
            packed_inputs = partition_and_pack(
                cudf_to_pylibcudf_table(local_df.iloc[i : i + args.batch_sz]),
                columns_to_hash=columns_to_hash,
                num_partitions=args.total_nparts,
                stream=DEFAULT_STREAM,
                device_mr=mr,
            )
            shuffler.insert_chunks(packed_inputs)

        # Tell shuffler we are done adding data
        for pid in range(args.total_nparts):
            shuffler.insert_finished(pid)

        # Extract and check shuffled partitions
        while not shuffler.finished():
            partition_id = shuffler.wait_any()
            packed_chunks = shuffler.extract(partition_id)
            partition = unpack_and_concat(
                packed_chunks,
                stream=DEFAULT_STREAM,
                device_mr=mr,
            )
            assert_eq(
                pylibcudf_to_cudf_dataframe(partition, column_names=column_names),
                expected[partition_id],
                sort_rows="a",
            )

        shuffler.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RapidsMP Ray Shuffling Actor example ",
    )
    parser.add_argument("--nranks", type=int, default=1)
    parser.add_argument("--num_rows", type=int, default=100)
    parser.add_argument("--batch_sz", type=int, default=-1)
    parser.add_argument("--total_nparts", type=int, default=-1)
    args = parser.parse_args()

    ray.init()  # init ray with all resources

    # Create shufflling actors
    gpu_actors = setup_ray_ucxx_cluster(ShufflingActor, args.nranks)

    try:
        # run the ShufflingActor.run method remotely
        ray.get([actor.run.remote(args) for actor in gpu_actors])  # type: ignore

    finally:
        for actor in gpu_actors:
            ray.kill(actor)
