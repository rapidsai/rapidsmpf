# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffler integration for Dask Distributed clusters."""

from __future__ import annotations

import logging
import threading
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast

import ucxx._lib.libucxx as ucx_api
from distributed import get_client, get_worker, wait
from distributed.diagnostics.plugin import SchedulerPlugin

import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmpf.buffer.buffer import MemoryType
from rapidsmpf.buffer.resource import BufferResource, LimitAvailableMemory
from rapidsmpf.buffer.spill_collection import SpillCollection
from rapidsmpf.communicator.ucxx import barrier, get_root_ucxx_address, new_communicator
from rapidsmpf.config import (
    Optional,
    OptionalBytes,
    Options,
    get_environment_variables,
)
from rapidsmpf.integrations.dask import _compat
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.statistics import Statistics

if TYPE_CHECKING:
    from collections.abc import Sequence

    import distributed
    from distributed.scheduler import Scheduler, TaskState

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.shuffler import Shuffler


_dask_logger = logging.getLogger("distributed.worker")
DataFrameT = TypeVar("DataFrameT")


@dataclass
class DaskWorkerContext:
    """
    RapidsMPF specific attributes for a Dask worker.

    Attributes
    ----------
    lock
        The global worker lock. Must be acquired before accessing attributes
        that might be modified while the worker is running such as the shufflers.
    br
        The buffer resource used by the worker exclusively.
    progress_thread
        The progress thread used by the worker.
    comm
        The UCXX communicator connected to all other workers in the Dask cluster.
    spill_collection
        A collection of Python objects that can be spilled to free up device memory.
    statistics
        The statistics used by the worker. If None, statistics is disabled.
    shufflers
        A mapping from shuffler IDs to active shuffler instances.
    options
        Configuration options.
    """

    lock: ClassVar[threading.RLock] = threading.RLock()
    br: BufferResource | None = None
    progress_thread: ProgressThread | None = None
    comm: Communicator | None = None
    spill_collection: SpillCollection = field(default_factory=SpillCollection)
    statistics: Statistics | None = None
    shufflers: dict[int, Shuffler] = field(default_factory=dict)
    options: Options = field(default_factory=Options)


def get_worker_context(
    dask_worker: distributed.Worker | None = None,
) -> DaskWorkerContext:
    """
    Retrieve the `DaskWorkerContext` associated with a Dask worker.

    If the worker context does not already exist on the worker, it will be created
    and attached to the worker under the attribute `_rapidsmpf_worker_context`.

    Parameters
    ----------
    dask_worker
        An optional Dask worker instance. If not provided, the current worker
        is retrieved using `get_worker()`.

    Returns
    -------
    The existing or newly initialized worker context.
    """
    with DaskWorkerContext.lock:
        dask_worker = dask_worker or get_worker()
        if not hasattr(dask_worker, "_rapidsmpf_worker_context"):
            dask_worker._rapidsmpf_worker_context = DaskWorkerContext()
        return cast(DaskWorkerContext, dask_worker._rapidsmpf_worker_context)


def get_worker_rank(dask_worker: distributed.Worker | None = None) -> int:
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


def global_rmpf_barrier(dependencies: Sequence[None]) -> None:
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
    ctx = get_worker_context()
    ctx.comm = new_communicator(n_ranks, None, None, options)
    ctx.comm.logger.trace(f"Rank {ctx.comm.rank} created")
    return get_root_ucxx_address(ctx.comm)


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
    ctx = get_worker_context()
    if ctx.comm is None:
        root_address = ucx_api.UCXAddress.create_from_buffer(root_address_bytes)
        ctx.comm = new_communicator(n_ranks, None, root_address, options)
        ctx.comm.logger.trace(f"Rank {ctx.comm.rank} created")

    ctx.comm.logger.trace(f"Rank {ctx.comm.rank} setup barrier")
    barrier(ctx.comm)
    ctx.comm.logger.trace(f"Rank {ctx.comm.rank} setup barrier passed")


def rmpf_worker_setup(
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
    ctx = get_worker_context(dask_worker)
    with ctx.lock:
        ctx.options = options

        # Insert RMM resource adaptor on top of the current RMM resource stack.
        mr = RmmResourceAdaptor(
            upstream_mr=rmm.mr.get_current_device_resource(),
            fallback_mr=(
                # Use a managed memory resource if OOM protection is enabled.
                rmm.mr.ManagedMemoryResource()
                if ctx.options.get_or_default(
                    "dask_oom_protection", default_value=False
                )
                else None
            ),
        )
        rmm.mr.set_current_device_resource(mr)

        # Print statistics at worker shutdown.
        if ctx.options.get_or_default("dask_statistics", default_value=False):
            ctx.statistics = Statistics(enable=True, mr=mr)
            weakref.finalize(
                dask_worker,
                lambda name, stats: print(name, stats.report()),
                name=str(dask_worker),
                stats=ctx.statistics,
            )

        assert ctx.comm is not None
        ctx.progress_thread = ProgressThread(ctx.comm, ctx.statistics)

        # Create a buffer resource with a limiting availability function.
        total_memory = rmm.mr.available_device_memory()[1]
        spill_device = ctx.options.get_or_default(
            "dask_spill_device", default_value=0.50
        )
        memory_available = {
            MemoryType.DEVICE: LimitAvailableMemory(
                mr, limit=int(total_memory * spill_device)
            )
        }
        ctx.br = BufferResource(
            mr,
            memory_available=memory_available,
            periodic_spill_check=ctx.options.get_or_default(
                "dask_periodic_spill_check", default_value=Optional(1e-3)
            ).value,
        )

        # If enabled, create a staging device buffer for the spilling to reduce
        # device memory pressure.
        # TODO: maybe have a pool of staging buffers?
        spill_staging_buffer_size = ctx.options.get_or_default(
            "dask_staging_spill_buffer",
            default_value=OptionalBytes("128 MiB"),
        ).value
        spill_staging_buffer = (
            None
            if spill_staging_buffer_size is None
            else rmm.DeviceBuffer(
                size=spill_staging_buffer_size, stream=DEFAULT_STREAM, mr=mr
            )
        )
        spill_staging_buffer_lock = threading.Lock()

        # Create a spill function that spills the python objects in the spill-
        # collection. This way, we have a central place (the dask worker) to track
        # and trigger spilling of python objects.
        def spill_func(amount: int) -> int:
            """
            Spill a specified amount of data from the Python object spill collection.

            This function attempts to use a preallocated staging device buffer to
            spill Python objects from the spill collection. If the staging buffer
            is currently in use, it will fall back to spilling without it.

            Parameters
            ----------
            amount
                The amount of data to spill, in bytes.

            Returns
            -------
            The actual amount of data spilled, in bytes.
            """
            if spill_staging_buffer is not None and spill_staging_buffer_lock.acquire(
                blocking=False
            ):
                try:
                    return ctx.spill_collection.spill(
                        amount,
                        stream=DEFAULT_STREAM,
                        device_mr=mr,
                        staging_device_buffer=spill_staging_buffer,
                    )
                finally:
                    spill_staging_buffer_lock.release()
            return ctx.spill_collection.spill(
                amount, stream=DEFAULT_STREAM, device_mr=mr
            )

        # Add the spill function using a negative priority (-10) such that spilling
        # of internal shuffle buffers (non-python objects) have higher priority than
        # spilling of the Python objects in the collection.
        ctx.br.spill_manager.add_spill_function(func=spill_func, priority=-10)


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
        rmpf_worker_setup,
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
