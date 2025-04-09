# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffler integration for Dask Distributed clusters."""

from __future__ import annotations

import logging
import threading
import weakref
from typing import TYPE_CHECKING, Any, TypeVar

import ucxx._lib.libucxx as ucx_api
from distributed import get_client, get_worker, wait
from distributed.diagnostics.plugin import SchedulerPlugin

import rmm.mr

from rapidsmp.buffer.buffer import MemoryType
from rapidsmp.buffer.resource import BufferResource, LimitAvailableMemory
from rapidsmp.buffer.spill_collection import SpillCollection
from rapidsmp.communicator.communicator import Communicator
from rapidsmp.communicator.ucxx import barrier, get_root_ucxx_address, new_communicator
from rapidsmp.progress_thread import ProgressThread
from rapidsmp.statistics import Statistics

if TYPE_CHECKING:
    from collections.abc import Sequence

    from distributed import Client, Worker
    from distributed.scheduler import Scheduler, TaskState


_dask_logger = logging.getLogger("distributed.worker")
DataFrameT = TypeVar("DataFrameT")


_worker_thread_lock: threading.RLock = threading.RLock()


def get_worker_thread_lock() -> threading.RLock:
    """
    Return the worker thread lock.

    Guard access to a dask worker's `_rmp_shufflers` attribute.

    Returns
    -------
    The work thread lock.
    """
    return _worker_thread_lock


def get_worker_rank(dask_worker: Worker | None = None) -> int:
    """
    Get the UCXX-comm rank for a Dask worker.

    Parameters
    ----------
    dask_worker
        Local Dask worker.

    Returns
    -------
    Local rapidsmp worker rank.

    Notes
    -----
    This function is expected to run on a Dask worker.
    """
    dask_worker = dask_worker or get_worker()
    return get_comm(dask_worker).rank


def global_rmp_barrier(dependencies: Sequence[None]) -> None:
    """
    Global barrier for rapidsmp shuffle.

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


async def rapidsmp_ucxx_rank_setup_root(n_ranks: int) -> bytes:
    """
    Set up the UCXX comm for the root worker.

    Parameters
    ----------
    n_ranks
        Number of ranks in the cluster / UCXX comm.

    Returns
    -------
    bytes
        The UCXX address of the root node.
    """
    dask_worker = get_worker()

    comm = new_communicator(n_ranks, None, None)
    comm.logger.trace(f"Rank {comm.rank} created")
    dask_worker._rapidsmp_comm = comm
    return get_root_ucxx_address(comm)


async def rapidsmp_ucxx_rank_setup_node(
    n_ranks: int, root_address_bytes: bytes
) -> None:
    """
    Set up the UCXX comms for a Dask worker.

    Parameters
    ----------
    n_ranks
        Number of ranks in the cluster / UCXX comm.
    root_address_bytes
        The UCXX address of the root node.
    """
    dask_worker = get_worker()

    if hasattr(dask_worker, "_rapidsmp_comm"):
        assert isinstance(dask_worker._rapidsmp_comm, Communicator)
        comm = dask_worker._rapidsmp_comm
    else:
        root_address = ucx_api.UCXAddress.create_from_buffer(root_address_bytes)
        comm = new_communicator(n_ranks, None, root_address)

        comm.logger.trace(f"Rank {comm.rank} created")
        dask_worker._rapidsmp_comm = comm

    comm.logger.trace(f"Rank {comm.rank} setup barrier")
    barrier(comm)
    comm.logger.trace(f"Rank {comm.rank} setup barrier passed")
    return None


def rmp_worker_setup(
    dask_worker: Worker,
    *,
    spill_device: float = 0.50,
    periodic_spill_check: float | None = 0.01,
    enable_statistics: bool = True,
) -> None:
    """
    Attach rapidsmp shuffling attributes to a Dask worker.

    Parameters
    ----------
    dask_worker
        The current Dask worker.
    spill_device
        GPU memory limit for shuffling.
    periodic_spill_check
        Enable periodic spill checks. A dedicated thread continuously checks
        and perform spilling based on the current available memory as reported
        by the buffer resource. The value of `periodic_spill_check` is used as
        the pause between checks (in seconds). If None, no periodic spill check
        is performed.
    enable_statistics
        Whether to track shuffler statistics.

    Warnings
    --------
    This function creates a new RMM memory pool, and
    sets it as the current device resource.

    See Also
    --------
    bootstrap_dask_cluster
        Setup a Dask cluster for rapidsmp shuffling.

    Notes
    -----
    This function is expected to run on a Dask worker.
    """
    with get_worker_thread_lock():
        if hasattr(dask_worker, "_rmp_shufflers"):
            return  # Worker already initialized

        # We start with no active shufflers
        dask_worker._rmp_shufflers = {}

        # Print statistics at worker shutdown.
        if enable_statistics:
            dask_worker._rmp_statistics = Statistics(enable=True)
            weakref.finalize(
                dask_worker,
                lambda name, stats: print(name, stats.report()),
                name=str(dask_worker),
                stats=dask_worker._rmp_statistics,
            )
        else:
            dask_worker._rmp_statistics = None

        dask_worker._rapidsmp_progress_thread = ProgressThread(
            dask_worker._rapidsmp_comm, dask_worker._rmp_statistics
        )

        # Setup a buffer_resource.
        # Wrap the current RMM resource in statistics adaptor.
        mr = rmm.mr.StatisticsResourceAdaptor(rmm.mr.get_current_device_resource())
        rmm.mr.set_current_device_resource(mr)
        total_memory = rmm.mr.available_device_memory()[1]
        memory_available = {
            MemoryType.DEVICE: LimitAvailableMemory(
                mr, limit=int(total_memory * spill_device)
            )
        }
        dask_worker._rmp_buffer_resource = BufferResource(
            mr,
            memory_available=memory_available,
            periodic_spill_check=periodic_spill_check,
        )

        # Add a new spill collection to enable spilling of DataFrames. We use a
        # negative priority (-10) such that spilling within shufflers have
        # higher priority than spilling of DataFrames.
        dask_worker._rmp_spill_collection = SpillCollection()
        dask_worker._rmp_buffer_resource.spill_manager.add_spill_function(
            func=dask_worker._rmp_spill_collection.spill, priority=-10
        )


_initialized_clusters: set[str] = set()


def bootstrap_dask_cluster(
    client: Client,
    *,
    spill_device: float = 0.50,
    enable_statistics: bool = True,
) -> None:
    """
    Setup a Dask cluster for rapidsmp shuffling.

    Parameters
    ----------
    client
        The current Dask client.
    spill_device
        GPU memory limit for shuffling.
    enable_statistics
        Whether to track shuffler statistics.

    Notes
    -----
    This utility must be executed before rapidsmp shuffling can be used within a
    Dask cluster. This function is called automatically by
    `rapidsmp.integrations.dask.core.rapids_shuffle_graph`, but may be called
    manually to set things up before the first shuffle.

    Subsequent shuffles on the same cluster will reuse the resources established
    on the cluster by this function.

    All the workers reported by :meth:`distributed.Client.scheduler_info` will
    be used. Note that rapidsmp does not currently support adding or removing
    workers from the cluster.
    """
    if client.asynchronous:
        raise ValueError("Client must be synchronous")

    if client.id in _initialized_clusters:
        return

    # Scheduler stuff
    scheduler_plugin = RMPSchedulerPlugin()
    client.register_plugin(scheduler_plugin)

    workers = sorted(client.scheduler_info()["workers"])
    n_ranks = len(workers)

    # Set up the comms for the root worker
    root_address_bytes = client.submit(
        rapidsmp_ucxx_rank_setup_root,
        n_ranks=len(workers),
        workers=workers[0],
        pure=False,
    ).result()

    # Set up the entire ucxx cluster
    ucxx_setup_futures = [
        client.submit(
            rapidsmp_ucxx_rank_setup_node,
            n_ranks=n_ranks,
            root_address_bytes=root_address_bytes,
            workers=worker,
            pure=False,
        )
        for worker in workers
    ]
    wait(ucxx_setup_futures)

    # Finally, prepare the rapidsmp resources on top of the UCXX comms
    client.run(
        rmp_worker_setup,
        spill_device=spill_device,
        enable_statistics=enable_statistics,
    )

    # Only run the above steps once
    _initialized_clusters.add(client.id)


class RMPSchedulerPlugin(SchedulerPlugin):
    """
    RAPIDS-MP Scheduler Plugin.

    The plugin helps manage integration with the RAPIDS-MP
    shuffle service by making it possible for the client
    to inform the scheduler of tasks that must be
    constrained to specific workers.
    """

    scheduler: Scheduler
    _rmp_restricted_tasks: dict[str, str]

    def __init__(self) -> None:
        self._rmp_restricted_tasks = {}
        self.scheduler = None

    async def start(  # noqa: D102
        self, scheduler: Scheduler
    ) -> None:  # numpydoc ignore=GL08
        self.scheduler = scheduler
        self.scheduler.stream_handlers.update(
            {"rmp_add_restricted_tasks": self.rmp_add_restricted_tasks}
        )

    def rmp_add_restricted_tasks(self, *args: Any, **kwargs: Any) -> None:
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
            self._rmp_restricted_tasks[key] = worker

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
        if self._rmp_restricted_tasks:
            tasks = kwargs.pop("tasks", [])
            for key in tasks:
                ts: TaskState = self.scheduler.tasks[key]
                if key in self._rmp_restricted_tasks:
                    worker = self._rmp_restricted_tasks.pop(key)
                    self.scheduler.set_restrictions({ts.key: {worker}})


def get_dask_client() -> Client:
    """
    Get the current Dask client.

    Returns
    -------
    Current Dask client.
    """
    client = get_client()
    bootstrap_dask_cluster(client)  # Make sure the cluster supports rapidsmp
    return client


def get_comm(dask_worker: Worker | None = None) -> Communicator:
    """
    Get the RAPIDS-MP UCXX comm for a Dask worker.

    Parameters
    ----------
    dask_worker
        Local Dask worker.

    Returns
    -------
    Current rapidsmp communicator.

    Notes
    -----
    This function is expected to run on a Dask worker.
    """
    dask_worker = dask_worker or get_worker()
    assert isinstance(dask_worker._rapidsmp_comm, Communicator), (
        f"Expected Communicator, got {dask_worker._rapidsmp_comm}"
    )
    return dask_worker._rapidsmp_comm


def get_progress_thread(dask_worker: Worker | None = None) -> ProgressThread:
    """
    Get the RAPIDS-MP progress thread for a Dask worker.

    Parameters
    ----------
    dask_worker
        Local Dask worker.

    Returns
    -------
    Current rapidsmp progress thread.

    Notes
    -----
    This function is expected to run on a Dask worker.
    """
    dask_worker = dask_worker or get_worker()
    assert isinstance(dask_worker._rapidsmp_progress_thread, ProgressThread), (
        f"Expected ProgressThread, got {dask_worker._rapidsmp_progress_thread}"
    )
    return dask_worker._rapidsmp_progress_thread
