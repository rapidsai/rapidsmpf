# Copyright (c) 2025, NVIDIA CORPORATION.
"""General Dask-integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dask import config
from dask_cuda import LocalCUDACluster
from distributed import get_client, get_worker, wait
from distributed.diagnostics.plugin import SchedulerPlugin

import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmp.buffer.buffer import MemoryType
from rapidsmp.buffer.resource import BufferResource, LimitAvailableMemory
from rapidsmp.integrations.dask import rapidsmp_ucxx_rank_setup
from rapidsmp.shuffler import Shuffler

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from distributed import Client, Worker
    from distributed.scheduler import Scheduler, TaskState


_initialized_clusters: set[str] = set()
_shuffle_counter: int = 0


def get_dask_client() -> Client:
    """Return a valid Dask client."""
    client = get_client()

    # Check that ucxx comms are ready
    bootstrap_dask_cluster(client)

    return client


def get_shuffle_id() -> int:
    """Return the unique id for a new shuffle."""
    global _shuffle_counter  # noqa: PLW0603

    _shuffle_counter += 1
    return _shuffle_counter


def global_rmp_barrier(
    shuffle_ids: tuple[int, ...],
    insert_results: Sequence[pd.DataFrame],
):
    """Global RMP shuffle barrier."""
    return shuffle_ids


def worker_rmp_barrier(
    shuffle_ids: tuple[int, ...],
    partition_count: int,
    global_barrier: tuple[int, ...],
):
    """Worker-specific RMP shuffle barrier."""
    for shuffle_id in shuffle_ids:
        shuffler = get_shuffler(shuffle_id)
        for pid in range(partition_count):
            shuffler.insert_finished(pid)
    return global_barrier


def rmp_setup(
    dask_worker,
    *,
    pool_size: float = 0.50,
    spill_device: float = 0.125,
):
    """Attach general RAPIDS-MP attributes to the worker."""
    # Add empty list of active shufflers
    dask_worker._rmp_shufflers = {}

    # Setup a buffer_resource
    # Create a RMM stack with both a device pool and statistics.
    available_memory = rmm.mr.available_device_memory()[1]
    rmm_pool_size = int(available_memory * pool_size)
    rmm_pool_size = (rmm_pool_size // 256) * 256
    mr = rmm.mr.StatisticsResourceAdaptor(
        rmm.mr.PoolMemoryResource(
            rmm.mr.CudaMemoryResource(),
            initial_pool_size=rmm_pool_size,
            maximum_pool_size=rmm_pool_size,
        )
    )
    rmm.mr.set_current_device_resource(mr)
    rmp_spill_device = int(available_memory * spill_device)
    rmp_spill_device = (rmp_spill_device // 256) * 256
    memory_available = {
        MemoryType.DEVICE: LimitAvailableMemory(mr, limit=rmp_spill_device)
    }
    dask_worker._memory_resource = mr
    dask_worker._buffer_resource = BufferResource(mr, memory_available)


def rapidsmp_ucxx_comm_setup_sync(client: Client):
    """
    Setup UCXX-based communicator across the Distributed cluster.

    Parameters
    ----------
    client: Client
        Distributed client connected to a Distributed cluster from
        which to setup the cluster.
    """
    workers = list(client.scheduler_info()["workers"])

    root_rank = [workers[0]]

    root_address_str = client.submit(
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
    wait(futures)


def bootstrap_dask_cluster(client: Client):
    """Setup a Dask cluster for rapidsmp shuffling."""

    def rmp_plugin_not_registered(dask_scheduler=None) -> bool:
        for plugin in dask_scheduler.plugins:
            if "RMPSchedulerPlugin" in plugin:
                return False
        return True

    if client.id not in _initialized_clusters:
        # Check that RMPSchedulerPlugin is registered
        if client.run_on_scheduler(rmp_plugin_not_registered):
            raise RuntimeError("RMPSchedulerPlugin was not found on the scheduler.")

        # Setup ucxx comms for rapidsmp
        rapidsmp_ucxx_comm_setup_sync(client)

        # Setup other rapidsmp requirements
        client.run(rmp_setup)

        _initialized_clusters.add(client.id)


class RMPSchedulerPlugin(SchedulerPlugin):
    """
    RAPIDS-MP Scheduler Plugin.

    The plugin helps manage integration with the RAPIDS-MP
    shuffle service by making it possible for the client
    to inform the scheduler of tasks that must be constrained
    to specific workers.
    """

    scheduler: Scheduler
    _rmp_restricted_tasks: dict[str, str]

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.scheduler.stream_handlers.update(
            {"rmp_add_restricted_tasks": self.rmp_add_restricted_tasks}
        )
        self.scheduler.add_plugin(self, name="rampidsmp_manger")
        self._rmp_restricted_tasks = {}

    def rmp_add_restricted_tasks(self, *args, **kwargs) -> None:
        """Add restricted tasks."""
        tasks = kwargs.pop("tasks", ())
        for key, worker in tasks.items():
            self._rmp_restricted_tasks[key] = worker

    def update_graph(self, *args, **kwargs) -> None:
        """Update graph on scheduler."""
        if self._rmp_restricted_tasks:
            tasks = kwargs.pop("tasks", [])
            for key in tasks:
                ts: TaskState = self.scheduler.tasks[key]
                if key in self._rmp_restricted_tasks:
                    worker = self._rmp_restricted_tasks.pop(key)
                    self._set_restriction(ts, worker)

    def _set_restriction(self, ts: TaskState, worker: str) -> None:
        if ts.annotations and "shuffle_original_restrictions" in ts.annotations:
            # This may occur if multiple barriers share the same output task,
            # e.g. in a hash join.
            return
        if ts.annotations is None:
            ts.annotations = {}
        ts.annotations["shuffle_original_restrictions"] = (
            ts.worker_restrictions.copy()
            if ts.worker_restrictions is not None
            else None
        )
        self.scheduler.set_restrictions({ts.key: {worker}})


def get_comm(worker: Worker | None = None):
    """Get the RAPIDS-MP UCXX comm for this worker."""
    worker = worker or get_worker()
    return worker._rapidsmp_comm


def get_worker_rank(dask_worker: Worker):
    """Get the UCXX-comm rank for this worker."""
    return get_comm(dask_worker).rank


def get_buffer_resource(worker: Worker | None = None):
    """Get the RAPIDS-MP buffer resource for this worker."""
    worker = worker or get_worker()
    return worker._buffer_resource


def get_shuffler(
    shuffle_id: int,
    partition_count: int | None = None,
    worker: Worker | None = None,
):
    """Get the active RAPIDS-MP shuffler."""
    worker = worker or get_worker()

    if shuffle_id not in worker._rmp_shufflers:
        if partition_count is None:
            raise RuntimeError(
                "Need partition_count to create new shuffler."
                f" shuffle_id: {shuffle_id}\n"
                f" Shufflers: {worker._rmp_shufflers}"
            )
        worker._rmp_shufflers[shuffle_id] = Shuffler(
            get_comm(worker),
            op_id=shuffle_id,
            total_num_partitions=partition_count,
            stream=DEFAULT_STREAM,
            br=get_buffer_resource(worker),
        )

    return worker._rmp_shufflers[shuffle_id]


def stage_shuffler(
    shuffle_id: int,
    partition_count: int,
    dask_worker: Worker,
):
    """Stage a shuffler on a Dask worker."""
    get_shuffler(
        shuffle_id,
        partition_count=partition_count,
        worker=dask_worker,
    )


def get_memory_resource(worker: Worker | None = None):
    """Get the RAPIDS-MP UCXX comm for this worker."""
    worker = worker or get_worker()
    return worker._memory_resource


class _LocalRMPCluster(LocalCUDACluster):
    """Local RAPIDSMP Dask cluster."""

    def __init__(self, **kwargs):
        self._rmp_shuffle_counter = 0
        preloads = config.get("distributed.scheduler.preload")
        preloads.append("rapidsmp.examples.dask")
        config.set({"distributed.scheduler.preload": preloads})
        super().__init__(**kwargs)


local_cuda_cluster = _LocalRMPCluster


def dask_setup(scheduler):
    """Setup dask cluster."""
    plugin = RMPSchedulerPlugin(scheduler)
    scheduler.add_plugin(plugin)
