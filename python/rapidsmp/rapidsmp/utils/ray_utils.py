# Copyright (c) 2025, NVIDIA CORPORATION.
"""Utils for the Ray integration."""

import math
from typing import TYPE_CHECKING

import numpy as np
import ray
import ucxx._lib.libucxx as ucx_api
from ray.actor import ActorClass

import cudf
import rmm

from rapidsmp.buffer.resource import BufferResource
from rapidsmp.communicator.ucxx import barrier, get_root_ucxx_address, new_communicator
from rapidsmp.shuffler import Shuffler, partition_and_pack, unpack_and_concat
from rapidsmp.testing import assert_eq
from rapidsmp.utils.cudf import (
    cudf_to_pylibcudf_table,
    pylibcudf_to_cudf_dataframe,
)
from rapidsmp.integrations.ray import     RapidsMPActor

if TYPE_CHECKING:
    from rapidsmp.communicator.communicator import Communicator

class ShufflingActor(RapidsMPActor):
    """Ray actor that performs a shuffle operation."""

    def __init__(self, nranks, num_rows=100, batch_size=-1, total_nparts=-1):
        super().__init__(nranks)
        self.num_rows = num_rows
        self.batch_size = batch_size
        self.total_nparts = total_nparts

    def _setup_device_mr(self):
        mr = rmm.mr.CudaMemoryResource()
        rmm.mr.set_current_device_resource(mr)
        return mr

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
        if self.comm is None:
            raise RuntimeError("RapidsMP not initialized")

        # If DEFAULT_STREAM was imported outside of this context, it will be pickled,
        # and it is not serializable. Therefore, we need to import it here.
        from rmm.pylibrmm.stream import DEFAULT_STREAM

        # if total_nparts is not set, set it to the number of ranks
        self.total_nparts = self._nranks if self.total_nparts < 0 else self.total_nparts

        mr = self._setup_device_mr()
        br = BufferResource(mr)

        df = self._gen_cudf()
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
                num_partitions=self.total_nparts,
                stream=DEFAULT_STREAM,
                device_mr=mr,
            ).items()
        }

        # Create shuffler
        shuffler = Shuffler(
            self.comm,
            op_id=0,
            total_num_partitions=self.total_nparts,
            stream=DEFAULT_STREAM,
            br=br,
        )

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
                device_mr=mr,
            )
            shuffler.insert_chunks(packed_inputs)

        # Tell shuffler we are done adding data
        for pid in range(self.total_nparts):
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