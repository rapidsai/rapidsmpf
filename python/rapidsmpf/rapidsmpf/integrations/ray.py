# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Integration for Ray clusters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ray
import ucxx._lib.libucxx as ucx_api
from ray.actor import ActorClass

from rapidsmpf.communicator.ucxx import barrier, get_root_ucxx_address, new_communicator
from rapidsmpf.config import Options, get_environment_variables

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator


class RapidsMPFActor:
    """
    RapidsMPFActor is a base class that instantiates a UCXX communication within them.

    Parameters
    ----------
    nranks
        The number of workers in the cluster.

    Examples
    --------
    >>> @ray.remote(num_cpus=1)
    ... class DummyActor(RapidsMPFActor): ...
    >>> actors = setup_ray_ucx_cluster(DummyActor, 2)
    >>> ray.get([actor.status_check.remote() for actor in actors]
    """

    def __init__(self, nranks: int):
        self._rank: int = -1
        self._nranks: int = nranks
        self._comm: Communicator | None = None

    def setup_root(self) -> tuple[int, bytes]:
        """
        Setup root communicator in the cluster.

        Returns
        -------
        rank
            The rank of the root.
        root_address_bytes
            The address of the root.
        """
        self._comm = new_communicator(
            self._nranks, None, None, Options(get_environment_variables())
        )
        self._rank = self._comm.rank
        self._comm.logger.trace(f"Rank {self._rank} created as root")
        return self._rank, get_root_ucxx_address(self._comm)

    def setup_worker(self, root_address_bytes: bytes) -> None:
        """
        Setup the worker in the cluster once the root is initialized.

        This method needs to be called by every worker including the root.

        Parameters
        ----------
        root_address_bytes
            The address of the root.
        """
        if not self._comm:
            # this is not the root and a comm needs to be instantiated
            root_address = ucx_api.UCXAddress.create_from_buffer(root_address_bytes)
            # create a comm pointing to the root_address
            self._comm = new_communicator(
                self._nranks, None, root_address, Options(get_environment_variables())
            )
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
            return f"RapidsMPFActor(rank:{self._rank}, nranks:{self._nranks}, \
                                Communicator:{self._comm.get_str()})"
        else:
            raise RuntimeError("Communicator not initialized")

    def is_initialized(self) -> bool:
        """
        Check if the communicator is initialized.

        Returns
        -------
        True if the communicator is initialized, False otherwise.
        """
        return self._comm is not None and self._rank != -1

    def rank(self) -> int:
        """
        Get the rank of the worker, as inferred from the UCXX communicator.

        Returns
        -------
        The rank of the worker
        """
        return self._rank

    def nranks(self) -> int:
        """
        Get the number of ranks in the UCXX communicator.

        Returns
        -------
        The number of ranks in the UCXX communicator
        """
        return self._nranks

    @property
    def comm(self) -> Communicator:
        """
        The UCXX communicator object.

        Returns
        -------
        The UCXX communicator object if initialized, otherwise None

        Raises
        ------
        RuntimeError if the communicator is not initialized

        Notes
        -----
        This property is not meant to be called remotely from the client.
        Then Ray will attempt to serialize the Communicator object, which will fail.
        Instead, the subclasses can use the `comm` property to access the communicator.
        For example, to create a Shuffle operation
        """
        if self._comm is None:
            raise RuntimeError("Communicator not initialized")
        return self._comm


def setup_ray_ucxx_cluster(
    actor_cls: ray.actor.ActorClass, num_workers: int, *args: Any, **kwargs: Any
) -> list[ray.actor.ActorHandle]:
    """
    A utility method to setup the UCXX communication using RapidsMPFActor actor objects.

    Parameters
    ----------
    actor_cls
        The actor class to be instantiated in the cluster.
    num_workers
        The number of workers in the cluster.
    *args
        Additional arguments to be passed to the actor class.
    **kwargs
        Additional keyword arguments to be passed to the actor class.

    Returns
    -------
    gpu_actors
        A list of actors in the cluster.
    """
    # check if the actor_cls extends the ActorClass and RapidsMPFActor classes
    if not (
        issubclass(type(actor_cls), ActorClass)
        and issubclass(type(actor_cls), RapidsMPFActor)
    ):
        raise TypeError(
            "actor_cls is not a subclass of ray.actor.ActorClass and rapidsmpf.integrations.ray.RapidsMPFActor"
        )

    # initialize the actors remotely in the cluster
    gpu_actors = [
        actor_cls.remote(num_workers, *args, **kwargs) for _ in range(num_workers)
    ]

    # initialize the first actor as the root remotely in the cluster
    _, root_address_bytes = ray.get(gpu_actors[0].setup_root.remote())

    # setup the workers in the cluster with the root address
    ray.get([actor.setup_worker.remote(root_address_bytes) for actor in gpu_actors])

    return gpu_actors
