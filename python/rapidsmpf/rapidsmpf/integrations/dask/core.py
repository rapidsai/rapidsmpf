# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffler integration for Dask Distributed clusters."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import ucxx._lib.libucxx as ucx_api
from distributed import get_client, get_worker, wait
from distributed.diagnostics.plugin import SchedulerPlugin

from rapidsmpf.communicator.ucxx import barrier, get_root_ucxx_address, new_communicator
from rapidsmpf.config import (
    Options,
    get_environment_variables,
)
from rapidsmpf.integrations import WorkerContext
from rapidsmpf.integrations.core import rmpf_worker_setup
from rapidsmpf.integrations.dask import _compat

if TYPE_CHECKING:
    from collections.abc import Sequence

    import distributed
    from distributed.scheduler import Scheduler, TaskState


_dask_logger = logging.getLogger("distributed.worker")


def get_worker_context(
    worker: distributed.Worker | None = None,
) -> WorkerContext:
    """
    Retrieve the ``WorkerContext`` associated with a Dask worker.

    If the worker context does not already exist on the worker, it will be created.

    Parameters
    ----------
    worker
        An optional Dask worker instance. If not provided, the current worker
        is retrieved using ``get_worker()``.

    Returns
    -------
    The existing or newly initialized worker context.
    """
    with WorkerContext.lock:
        worker = worker or get_worker()
        return worker._rapidsmpf_worker_context  # type: ignore[no-any-return]


def get_dask_worker_rank(dask_worker: distributed.Worker | None = None) -> int:
    """
    Get the UCXX-comm rank for a Dask worker.

    Parameters
    ----------
    dask_worker
        Local Dask worker.

    Returns
    -------
    Local RapidsMPF worker rank.

    Notes
    -----
    This function is expected to run on a Dask worker.
    """
    comm = get_worker_context(dask_worker).comm
    assert comm is not None
    return comm.rank


def global_rmpf_barrier(*dependencies: Sequence[None]) -> None:
    """
    Global barrier for RapidsMPF shuffle.

    Parameters
    ----------
    dependencies
        Sequence of nulls, used to enforce barrier dependencies.

    Notes
    -----
    A global barrier task does NOT need to be restricted
    to a specific Dask worker.

    This function is meant to be a no-op.
    """


async def rapidsmpf_ucxx_rank_setup_root(n_ranks: int, options: Options) -> bytes:
    """
    Set up the UCXX comm for the root worker.

    Parameters
    ----------
    n_ranks
        Number of ranks in the cluster / UCXX comm.
    options
        Configuration options.

    Returns
    -------
    bytes
        The UCXX address of the root node.
    """
    with WorkerContext.lock:
        worker = get_worker()
        comm = new_communicator(n_ranks, None, None, options)
        worker._rapidsmpf_comm = comm
        comm.logger.trace(f"Rank {comm.rank} created")
        return get_root_ucxx_address(comm)


async def rapidsmpf_ucxx_rank_setup_node(
    n_ranks: int, root_address_bytes: bytes, options: Options
) -> None:
    """
    Set up the UCXX comms for a Dask worker.

    Parameters
    ----------
    n_ranks
        Number of ranks in the cluster / UCXX comm.
    root_address_bytes
        The UCXX address of the root node.
    options
        Configuration options.
    """
    with WorkerContext.lock:
        worker = get_worker()
        if not hasattr(worker, "_rapidsmpf_comm"):
            # Not the root rank
            root_address = ucx_api.UCXAddress.create_from_buffer(root_address_bytes)
            comm = new_communicator(n_ranks, None, root_address, options)
            worker._rapidsmpf_comm = comm
            comm.logger.trace(f"Rank {comm.rank} created")
        comm = worker._rapidsmpf_comm
        comm.logger.trace(f"Rank {comm.rank} setup barrier")
        barrier(comm)
        comm.logger.trace(f"Rank {comm.rank} setup barrier passed")


def dask_worker_setup(
    dask_worker: distributed.Worker,
    *,
    options: Options,
) -> None:
    """
    Attach RapidsMPF shuffling attributes to a Dask worker.

    Parameters
    ----------
    dask_worker
        The current Dask worker.
    options
        Configuration options.

    Warnings
    --------
    This function creates a new RMM memory pool, and
    sets it as the current device resource.

    See Also
    --------
    bootstrap_dask_cluster
        Setup a Dask cluster for RapidsMPF shuffling.

    Notes
    -----
    This function is expected to run on a Dask worker.
    """
    try:
        comm = dask_worker._rapidsmpf_comm
    except AttributeError:
        raise RuntimeError("Dask cluster not yet bootstrapped") from None
    with WorkerContext.lock:
        if not hasattr(dask_worker, "_rapidsmpf_worker_context"):
            dask_worker._rapidsmpf_worker_context = rmpf_worker_setup(
                dask_worker,
                "dask_",
                comm=comm,
                options=options,
            )


_initialized_clusters: set[str] = set()


def bootstrap_dask_cluster(
    client: distributed.Client,
    *,
    options: Options = Options(),
) -> None:
    """
    Setup a Dask cluster for RapidsMPF shuffling.

    Calling ``bootstrap_dask_cluster`` multiple times on the same worker is a
    noop, which also means that any new options values are ignored.

    Parameters
    ----------
    client
        The current Dask client.
    options
        Configuration options. Reads environment variables for any options not set
        explicitly using `get_environment_variables()`.

    Notes
    -----
    This utility must be executed before RapidsMPF shuffling can be used within a
    Dask cluster. This function is called automatically by
    `rapidsmpf.integrations.dask.rapidsmpf_shuffle_graph`, but may be called
    manually to set things up before the first shuffle.

    Subsequent shuffles on the same cluster will reuse the resources established
    on the cluster by this function.

    All the workers reported by :meth:`distributed.Client.scheduler_info` will
    be used. Note that RapidsMPF does not currently support adding or removing
    workers from the cluster.
    """
    if client.asynchronous:
        raise ValueError("Client must be synchronous")

    if client.id in _initialized_clusters:
        return

    # Scheduler stuff
    scheduler_plugin = RMPFSchedulerPlugin()
    client.register_plugin(scheduler_plugin)

    kwargs = {}
    if _compat.DISTRIBUTED_2025_4_0():
        kwargs["n_workers"] = -1
    workers = sorted(client.scheduler_info(**kwargs)["workers"])
    n_ranks = len(workers)

    # Insert missing config options from environment variables.
    options.insert_if_absent(get_environment_variables())
    # Set up the comms for the root worker
    root_address_bytes = client.submit(
        rapidsmpf_ucxx_rank_setup_root,
        n_ranks=len(workers),
        options=options,
        workers=workers[0],
        pure=False,
    ).result()

    # Set up the entire ucxx cluster
    ucxx_setup_futures = [
        client.submit(
            rapidsmpf_ucxx_rank_setup_node,
            n_ranks=n_ranks,
            root_address_bytes=root_address_bytes,
            options=options,
            workers=worker,
            pure=False,
        )
        for worker in workers
    ]
    wait(ucxx_setup_futures)

    # Finally, prepare the RapidsMPF resources on top of the UCXX comms
    client.run(
        dask_worker_setup,
        options=options,
    )

    # Only run the above steps once
    _initialized_clusters.add(client.id)


class RMPFSchedulerPlugin(SchedulerPlugin):
    """
    RapidsMPF Scheduler Plugin.

    The plugin helps manage integration with the RAPIDS-MPF
    shuffle service by making it possible for the client
    to inform the scheduler of tasks that must be
    constrained to specific workers.
    """

    scheduler: Scheduler
    _rmpf_restricted_tasks: dict[str, str]

    def __init__(self) -> None:
        self._rmpf_restricted_tasks = {}
        self.scheduler = None

    async def start(  # noqa: D102
        self, scheduler: Scheduler
    ) -> None:  # numpydoc ignore=GL08
        self.scheduler = scheduler
        self.scheduler.stream_handlers.update(
            {"rmpf_add_restricted_tasks": self.rmpf_add_restricted_tasks}
        )

    def rmpf_add_restricted_tasks(self, *args: Any, **kwargs: Any) -> None:
        """
        Add restricted tasks that must run on specific workers.

        Parameters
        ----------
        *args
            Positional arguments (ignored).
        **kwargs
            Key-word arguments. Used to pass dictionary of
            restricted tasks.
        """
        tasks = kwargs.pop("tasks", ())
        for key, worker in tasks.items():
            self._rmpf_restricted_tasks[key] = worker

    def update_graph(self, *args: Any, **kwargs: Any) -> None:
        """
        Graph update hook: apply task restrictions.

        Parameters
        ----------
        *args
            Positional arguments (ignored).
        **kwargs
            Key-word arguments. Used to access new tasks.
        """
        if self._rmpf_restricted_tasks:
            tasks = kwargs.pop("tasks", [])
            for key in tasks:
                ts: TaskState = self.scheduler.tasks[key]
                if key in self._rmpf_restricted_tasks:
                    worker = self._rmpf_restricted_tasks.pop(key)
                    self.scheduler.set_restrictions({ts.key: {worker}})


def get_dask_client(options: Options = Options()) -> distributed.Client:
    """
    Get the current Dask client.

    options
        Configuration options.

    Returns
    -------
    Current Dask client.
    """
    client = get_client()
    # Make sure the cluster supports RapidsMPF
    bootstrap_dask_cluster(client, options=options)
    return client
