# Copyright (c) 2025, NVIDIA CORPORATION.
"""Integration for Dask Distributed clusters."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import ucxx._lib.libucxx as ucx_api
from distributed import get_worker

from rapidsmp.communicator.ucxx import barrier, get_root_ucxx_address, new_communicator

if TYPE_CHECKING:
    from distributed import Client


async def rapidsmp_ucxx_rank_setup(
    nranks: int, root_address_str: str | None = None
) -> str | None:
    """
    Setup UCXX-based communicator on single rank.

    This function should run in each Dask worker that is to be part of the RAPIDSMP cluster.

    First, this must run on the elected root rank and will then return the UCXX address
    of the root as a string.

    With the root rank already setup, this should run again with the valid root address
    specified via `root_address_str` in all workers, including the root rank. Non-root
    ranks will connect to the root and all ranks, including the root, will then run a
    barrier, the barrier is important to ensure the underlying UCXX worker is progressed,
    thus why it is necessary to run again on root.

    Parameters
    ----------
    nranks: int
        The total number of ranks requested for the cluster.
    root_address_str: str, optional
        The address of the root rank if it has been already setup, `None` if this is
        setting up the root rank. Note that this function must run twice on the root rank
        one to initialize it, and again to ensure synchronization with other ranks. See
        the function extended description for details.

    Returns
    -------
    root_address: str, optional
        Returns the root rank address as a string if this function was called to setup the
        root, otherwise returns `None`.
    """
    dask_worker = get_worker()

    if root_address_str is None:
        comm = new_communicator(nranks, None, None)
        comm.logger.trace(f"Rank {comm.rank} created")
        dask_worker._rapidsmp_comm = comm
        return get_root_ucxx_address(comm)
    else:
        if hasattr(dask_worker, "_rapidsmp_comm"):
            comm = dask_worker._rapidsmp_comm
        else:
            root_address = ucx_api.UCXAddress.create_from_buffer(root_address_str)
            comm = new_communicator(nranks, None, root_address)

            comm.logger.trace(f"Rank {comm.rank} created")
            dask_worker._rapidsmp_comm = comm

        comm.logger.trace(f"Rank {comm.rank} setup barrier")
        barrier(comm)
        comm.logger.trace(f"Rank {comm.rank} setup barrier passed")
        return None


async def rapidsmp_ucxx_comm_setup(client: Client):
    """
    Setup UCXX-based communicator across the Distributed cluster.

    Setup UCXX-based communicator across the Distributed cluster, maintaining the
    communicator alive via state stored in the Distributed workers.

    Parameters
    ----------
    client: Client
        Distributed client connected to a Distributed cluster from which to setup the
        cluster.
    """
    workers = list(client.scheduler_info()["workers"])

    root_rank = [workers[0]]

    root_address_str = await client.submit(
        rapidsmp_ucxx_rank_setup,
        nranks=len(workers),
        root_address_str=None,
        workers=root_rank,
        pure=False,
    ).result()

    futures = [
        client.submit(
            rapidsmp_ucxx_rank_setup,
            nranks=len(workers),
            root_address_str=root_address_str,
            workers=[w],
            pure=False,
        )
        for w in workers
    ]
    await asyncio.gather(*futures)
