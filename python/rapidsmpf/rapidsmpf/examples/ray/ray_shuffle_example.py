# Copyright (c) 2025, NVIDIA CORPORATION.
"""Example running a RapidsMPF Shuffle operation using Ray and UCXX communication."""

from __future__ import annotations

import argparse
import math

import numpy as np
import ray

import cudf
import rmm

from rapidsmpf.integrations.cudf.partition import (
    partition_and_pack,
    unpack_and_concat,
)
from rapidsmpf.integrations.ray import setup_ray_ucxx_cluster
from rapidsmpf.testing import assert_eq
from rapidsmpf.utils.cudf import (
    cudf_to_pylibcudf_table,
    pylibcudf_to_cudf_dataframe,
)
from rapidsmpf.utils.ray_utils import BaseShufflingActor


class ShufflingActor(BaseShufflingActor):
    """
    An example of a Ray actor that performs a shuffle operation.

    It makes use of the BaseShufflingActor class to initiate a shuffler,
    and uses that for cudf dataframe example.

    Parameters
    ----------
    nranks
        Number of ranks.
    num_rows
        Number of rows in the input dataframe.
    batch_size
        Batch size (rows) of the input. The input dataframe will be split into batches of this size.
    total_nparts
        Total number of partitions into which the input dataframe will be partitioned.
    """

    def __init__(
        self,
        nranks: int,
        num_rows: int = 100,
        batch_size: int = -1,
        total_nparts: int = -1,
    ):
        super().__init__(nranks)
        self._num_rows: int = num_rows
        self._batch_size: int = batch_size
        self._total_nparts: int = total_nparts if total_nparts > 0 else nranks

    def _gen_cudf(self) -> cudf.DataFrame:
        """
        Generate a dummy dataframe.

        Returns
        -------
        cudf.DataFrame
            The input dataframe.
        """
        # Every rank creates the full input dataframe and all the expected partitions
        # (also partitions this rank might not get after the shuffle).

        np.random.seed(42)  # Make sure all ranks create the same input dataframe.

        return cudf.DataFrame(
            {
                "a": range(self._num_rows),
                "b": np.random.randint(0, 1000, self._num_rows),
                "c": ["cat", "dog"] * (self._num_rows // 2),
            }
        )

    def run(self) -> None:
        """Run the shuffle operation, and this will be called remotely from the client."""
        # If DEFAULT_STREAM was imported outside of this context, it will be pickled,
        # and it is not serializable. Therefore, we need to import it here.
        from rmm.pylibrmm.stream import DEFAULT_STREAM

        df = self._gen_cudf()
        columns_to_hash = (df.columns.get_loc("b"),)
        column_names = list(df.columns)

        mr = rmm.mr.get_current_device_resource()  # use the current device resource
        stream = DEFAULT_STREAM  # use the default stream

        # Calculate the expected output partitions on all ranks
        expected = {
            partition_id: pylibcudf_to_cudf_dataframe(
                unpack_and_concat(
                    [packed],
                    stream=stream,
                    device_mr=mr,
                ),
                column_names=column_names,
            )
            for partition_id, packed in partition_and_pack(
                cudf_to_pylibcudf_table(df),
                columns_to_hash=columns_to_hash,
                num_partitions=self._total_nparts,
                stream=stream,
                device_mr=mr,
            ).items()
        }

        # initialize a shuffler with the default buffer resource
        shuffler = self.create_shuffler(
            0, total_num_partitions=self._total_nparts, stream=stream
        )

        # Slice df and submit local slices to shuffler
        stride = math.ceil(self._num_rows / self.comm.nranks)
        local_df = df.iloc[self.comm.rank * stride : (self.comm.rank + 1) * stride]
        num_rows_local = len(local_df)
        self._batch_size = num_rows_local if self._batch_size < 0 else self._batch_size
        for i in range(0, num_rows_local, self._batch_size):
            packed_inputs = partition_and_pack(
                cudf_to_pylibcudf_table(local_df.iloc[i : i + self._batch_size]),
                columns_to_hash=columns_to_hash,
                num_partitions=self._total_nparts,
                stream=stream,
                device_mr=mr,
            )
            shuffler.insert_chunks(packed_inputs)

        # Tell shuffler we are done adding data
        for pid in range(self._total_nparts):
            shuffler.insert_finished(pid)

        # Extract and check shuffled partitions
        while not shuffler.finished():
            partition_id = shuffler.wait_any()
            packed_chunks = shuffler.extract(partition_id)
            partition = unpack_and_concat(
                packed_chunks,
                stream=stream,
                device_mr=mr,
            )
            assert_eq(
                pylibcudf_to_cudf_dataframe(partition, column_names=column_names),
                expected[partition_id],
                sort_rows="a",
            )

        shuffler.shutdown()


@ray.remote(num_gpus=1)
class GpuShufflingActor(ShufflingActor):
    """Shuffle example class with 1 GPU resource."""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RapidsMPF Ray Shuffling Actor example ",
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
