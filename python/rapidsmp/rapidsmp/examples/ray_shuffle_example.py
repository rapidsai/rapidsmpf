# Copyright (c) 2025, NVIDIA CORPORATION.
"""Example running a RapidsMP Shuffle operation using Ray and UCXX communication."""

from __future__ import annotations

import argparse
import math

import numpy as np
import ray

import cudf

from rapidsmp.integrations.ray import setup_ray_ucxx_cluster
from rapidsmp.shuffler import partition_and_pack, unpack_and_concat
from rapidsmp.testing import assert_eq
from rapidsmp.utils.cudf import (
    cudf_to_pylibcudf_table,
    pylibcudf_to_cudf_dataframe,
)
from rapidsmp.utils.ray_utils import AbstractShufflingActor


class ShufflingActor(AbstractShufflingActor):
    """Ray actor that performs a shuffle operation."""

    def __init__(self, nranks, num_rows=100, batch_size=-1, total_nparts=-1):
        super().__init__(nranks, op_id=0, total_nparts=total_nparts)
        self.num_rows = num_rows
        self.batch_size = batch_size

    def _gen_cudf(self):
        # Every rank creates the full input dataframe and all the expected partitions
        # (also partitions this rank might not get after the shuffle).

        np.random.seed(42)  # Make sure all ranks create the same input dataframe.

        return cudf.DataFrame(
            {
                "a": range(self.num_rows),
                "b": np.random.randint(0, 1000, self.num_rows),
                "c": ["cat", "dog"] * (self.num_rows // 2),
            }
        )

    def run(self) -> None:
        """Runs the shuffle operation, and this will be called remotely from the client."""
        # If DEFAULT_STREAM was imported outside of this context, it will be pickled,
        # and it is not serializable. Therefore, we need to import it here.
        from rmm.pylibrmm.stream import DEFAULT_STREAM

        df = self._gen_cudf()
        columns_to_hash = (df.columns.get_loc("b"),)
        column_names = list(df.columns)

        # Calculate the expected output partitions on all ranks
        expected = {
            partition_id: pylibcudf_to_cudf_dataframe(
                unpack_and_concat(
                    [packed],
                    stream=DEFAULT_STREAM,
                    device_mr=self.device_mr,
                ),
                column_names=column_names,
            )
            for partition_id, packed in partition_and_pack(
                cudf_to_pylibcudf_table(df),
                columns_to_hash=columns_to_hash,
                num_partitions=self.total_nparts,
                stream=DEFAULT_STREAM,
                device_mr=self.device_mr,
            ).items()
        }

        # Slice df and submit local slices to shuffler
        stride = math.ceil(self.num_rows / self.comm.nranks)
        local_df = df.iloc[self.comm.rank * stride : (self.comm.rank + 1) * stride]
        num_rows_local = len(local_df)
        self.batch_size = num_rows_local if self.batch_size < 0 else self.batch_size
        for i in range(0, num_rows_local, self.batch_size):
            packed_inputs = partition_and_pack(
                cudf_to_pylibcudf_table(local_df.iloc[i : i + self.batch_size]),
                columns_to_hash=columns_to_hash,
                num_partitions=self.total_nparts,
                stream=DEFAULT_STREAM,
                device_mr=self.device_mr,
            )
            self.shuffler.insert_chunks(packed_inputs)

        # Tell shuffler we are done adding data
        for pid in range(self.total_nparts):
            self.shuffler.insert_finished(pid)

        # Extract and check shuffled partitions
        while not self.shuffler.finished():
            partition_id = self.shuffler.wait_any()
            packed_chunks = self.shuffler.extract(partition_id)
            partition = unpack_and_concat(
                packed_chunks,
                stream=DEFAULT_STREAM,
                device_mr=self.device_mr,
            )
            assert_eq(
                pylibcudf_to_cudf_dataframe(partition, column_names=column_names),
                expected[partition_id],
                sort_rows="a",
            )

        self.shutdown()


@ray.remote(num_gpus=1)
class GpuShufflingActor(ShufflingActor):
    """Shuffle example class with 1 GPU resource."""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RapidsMP Ray Shuffling Actor example ",
    )
    parser.add_argument("--nranks", type=int, default=1)
    parser.add_argument("--num_rows", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--total_nparts", type=int, default=-1)
    args = parser.parse_args()

    ray.init()  # init ray with all resources

    # Create shufflling actors
    gpu_actors = setup_ray_ucxx_cluster(
        GpuShufflingActor,
        args.nranks,
        args.num_rows,
        args.batch_size,
        args.total_nparts,
    )

    try:
        # run the ShufflingActor.run method remotely
        ray.get([actor.run.remote() for actor in gpu_actors])  # type: ignore

    finally:
        for actor in gpu_actors:
            ray.kill(actor)
