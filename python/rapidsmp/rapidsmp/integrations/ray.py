# Copyright (c) 2025, NVIDIA CORPORATION.
"""Integration for Ray clusters."""

from __future__ import annotations

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

if TYPE_CHECKING:
    from rapidsmp.communicator.communicator import Communicator


class RapidsMPActor:
    """
    RapidsMPActor is a base class that instantiates a UCXX communication within them.

    Example:
    >>> @ray.remote(num_cpus=1)
    ... class DummyActor(RapidsMPActor): ...
    >>> actors = setup_ray_ucx_cluster(DummyActor, 2)
    >>> ray.get([actor.status_check.remote() for actor in actors])

    """

    def __init__(self, nranks: int):
        """
        Initialize the RapidsMPActor.

        Parameters
        ----------
        nranks
            The number of workers in the cluster
        """
        self._rank: int = -1
        self._nranks: int = nranks
        self._comm: Communicator | None = None

    def setup_root(self) -> tuple[int, str]:
        """
        Setup root communicator in the cluster.

        Returns
        -------
        rank
            The rank of the root
        root_address_str
            The address of the root
        """
        self._comm = new_communicator(self._nranks, None, None)
        self._rank = self._comm.rank
        self._comm.logger.trace(f"Rank {self._rank} created as root")
        return self._rank, get_root_ucxx_address(self._comm)

    def setup_worker(self, root_address_str: str) -> None:
        """
        Setup the worker in the cluster once the root is initialized.

        This method needs to be called by every worker including the root.

        Parameters
        ----------
        root_address_str
            The address of the root
        """
        if not self._comm:
            # this is not the root and a comm needs to be instantiated
            root_address = ucx_api.UCXAddress.create_from_buffer(root_address_str)
            # create a comm pointing to the root_address
            self._comm = new_communicator(self._nranks, None, root_address)
            self._rank = self._comm.rank
            self._comm.logger.trace(f"Rank {self._rank} created")

        self._comm.logger.trace(f"Rank {self._rank} setup barrier")
        barrier(self._comm)
        self._comm.logger.trace(f"Rank {self._rank} setup barrier passed")

        if self._nranks != self._comm.nranks:
            raise RuntimeError(
                f"Number of ranks mismatch in the communicator: \
                               {self._nranks} != {self._comm.nranks}"
            )

    def to_string(self) -> str:
        """
        Return a string representation of the actor.

        Returns
        -------
            A string representation of the actor

        Raises
        ------
            RuntimeError if the communicator is not initialized
        """
        if self._comm:
            return f"RapidsMPActor(rank:{self._rank}, nranks:{self._nranks}, \
                                Communicator:{self._comm.get_str()})"
        else:
            raise RuntimeError("Communicator not initialized")

    def is_initialized(self) -> bool:
        """Check if the communicator is initialized."""
        return self._comm is not None and self._rank != -1

    def rank(self) -> int:
        """Get the rank of the worker, as inferred from the UCXX communicator."""
        return self._rank

    def nranks(self) -> int:
        """Get the number of ranks in the UCXX communicator."""
        return self._nranks

    @property
    def comm(self) -> Communicator | None:
        """
        The UCXX communicator object.

        Note: This property is not meant to be called remotely from the client.
        Then Ray will attempt to serialize the Communicator object, which will fail.
        Instead, the subclasses can use the `comm` property to access the communicator.
        For example, to create a Shuffle operation

        Returns
        -------
            The UCXX communicator object if initialized, otherwise None
        """
        return self._comm


def setup_ray_ucxx_cluster(
    actor_cls: ray.actor.ActorClass, num_workers: int, *args, **kwargs
) -> list[object]:
    """
    A utility method to setup the UCXX communication using RapidsMPActor actor objects.

    Parameters
    ----------
    actor_cls
        The actor class to be instantiated in the cluster
    num_workers
        The number of workers in the cluster
    *args
        Additional arguments to be passed to the actor class
    **kwargs
        Additional keyword arguments to be passed to the actor class

    Returns
    -------
    gpu_actors
        A list of actors in the cluster
    """
    # check if the actor_cls extends the ActorClass and RapidsMPActor classes
    if not (
        issubclass(type(actor_cls), ActorClass)
        and issubclass(type(actor_cls), RapidsMPActor)
    ):
        raise TypeError(
            "actor_cls is not a subclass of ray.actor.ActorClass and rapidsmp.integrations.ray.RapidsMPActor"
        )

    # initialize the actors remotely in the cluster
    gpu_actors = [
        actor_cls.remote(num_workers, *args, **kwargs) for _ in range(num_workers)
    ]

    # initialize the first actor as the root remotely in the cluster
    _, root_address_str = ray.get(gpu_actors[0].setup_root.remote())

    # setup the workers in the cluster with the root address
    ray.get([actor.setup_worker.remote(root_address_str) for actor in gpu_actors])

    return gpu_actors


class ShufflingActor(RapidsMPActor):
    """Ray actor that performs a shuffle operation."""

    def __init__(self, nranks, num_rows=100, batch_sz=-1, total_nparts=-1):
        super().__init__(nranks)
        self.num_rows = num_rows
        self.batch_sz = batch_sz
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
        self.batch_sz = num_rows_local if self.batch_sz < 0 else self.batch_sz
        for i in range(0, num_rows_local, self.batch_sz):
            packed_inputs = partition_and_pack(
                cudf_to_pylibcudf_table(local_df.iloc[i : i + self.batch_sz]),
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
