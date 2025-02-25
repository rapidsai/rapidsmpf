# Copyright (c) 2025, NVIDIA CORPORATION.
"""Integration for Ray clusters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ray
import ucxx._lib.libucxx as ucx_api

from rapidsmp.communicator.ucxx import barrier, get_root_ucxx_address, new_communicator

if TYPE_CHECKING:
    from rapidsmp.communicator.communicator import Communicator


class RapidsMPActor:
    """
    RapidsMPActor is a base class that instantiates a ucxx communication within them.

    Parameters
    ----------
        rank
            The rank of the worker

        nranks
            The number of workers in the cluster

        comm
            The ucxx communicator

    Example:
        .. code-block:: python
            class DummyActor(RapidsMPActor):
                def __init__(self, rank, num_workers):
                    super().__init__(rank, num_workers)


            gpu_actors = setup_ray_ucx_cluster(DummyActor, num_workers)
            ray.get([actor.status_check.remote() for actor in gpu_actors])
    """

    def __init__(self, rank: int, nranks: int):
        """
        Initialize the RapidsMPActor.

        Parameters
        ----------
        rank
            The rank of the worker
        nranks
            The number of workers in the cluster
        """
        self.rank: int = rank
        self.nranks: int = nranks
        self.comm: Communicator | None = None

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
        self.comm = new_communicator(self.nranks, None, None)
        self.comm.logger.trace(f"Rank {self.rank} created as root")
        return self.rank, get_root_ucxx_address(self.comm)

    def setup_worker(self, root_address_str: str) -> None:
        """
        Setup the worker in the cluster once the root is initialized.

        Parameters
        ----------
        root_address_str
            The address of the root
        """
        if not self.comm:
            # this is not the root and a comm needs to be instantiated
            root_address = ucx_api.UCXAddress.create_from_buffer(root_address_str)
            # create a comm pointing to the root_address
            self.comm = new_communicator(self.nranks, None, root_address)
            self.comm.logger.trace(f"Rank {self.comm.rank} created")

        self.comm.logger.trace(f"Rank {self.comm.rank} setup barrier")
        barrier(self.comm)
        self.comm.logger.trace(f"Rank {self.comm.rank} setup barrier passed")

    def status_check(self) -> None:
        """Check if the communicator is initialized."""
        if self.comm:
            print(f"Communicator: {self.comm.get_str()}")
        else:
            raise RuntimeError("Communicator not initialized")

    def is_initialized(self) -> bool:
        """
        Check if the communicator is initialized.

        Returns
        -------
            True if the communicator is initialized, False otherwise
        """
        return self.comm is not None

    @property
    def communicator(self) -> Communicator | None:
        """
        Get the underlying ucxx communicator.

        Returns
        -------
            The underlying ucxx communicator
        """
        return self.comm


def setup_ray_ucx_cluster(
    actor_cls: object, num_workers: int, root: int = 0
) -> list[object]:
    """
    A utility method to setup the ucxx communication using RapidsMPActor actor objects.

    Parameters
    ----------
    actor_cls
        The actor class to be instantiated in the cluster
    num_workers
        The number of workers in the cluster
    root
        The rank of the root actor (default = 0)

    Returns
    -------
        gpu_actors
            A list of actors in the cluster
    """
    # check if the actor_cls has a remote method
    if (
        not hasattr(actor_cls, "remote")
        or not hasattr(actor_cls, "setup_root")
        or not hasattr(actor_cls, "setup_worker")
    ):
        raise ValueError(
            "actor_cls isn't a RayActor or it doesn't extend RapidsMPActor"
        )

    # initialize the actors remotely in the cluster
    gpu_actors = [actor_cls.remote(i, num_workers) for i in range(num_workers)]

    # initialize the root actor remotely in the cluster
    root_rank, root_address_str = ray.get(gpu_actors[root].setup_root.remote())
    if root_rank != root:
        raise RuntimeError("Root rank mismatch")

    # setup the workers in the cluster with the root address
    ray.get([actor.setup_worker.remote(root_address_str) for actor in gpu_actors])

    return gpu_actors
