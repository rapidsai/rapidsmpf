# Copyright (c) 2025, NVIDIA CORPORATION.
"""Integration for Dask Distributed clusters."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol

import ucxx._lib.libucxx as ucx_api
from dask import config
from dask_cuda import LocalCUDACluster
from distributed import get_client, get_worker, wait
from distributed.diagnostics.plugin import SchedulerPlugin

import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmp.buffer.buffer import MemoryType
from rapidsmp.buffer.resource import BufferResource, LimitAvailableMemory
from rapidsmp.communicator.ucxx import barrier, get_root_ucxx_address, new_communicator
from rapidsmp.shuffler import Shuffler

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping, Sequence

    from distributed import Client, Worker
    from distributed.scheduler import Scheduler, TaskState


_initialized_clusters: set[str] = set()
_shuffle_counter: int = 0


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
    nranks
        The total number of ranks requested for the cluster.
    root_address_str
        The address of the root rank if it has been already setup, `None` if this is
        setting up the root rank. Note that this function must run twice on the root rank
        one to initialize it, and again to ensure synchronization with other ranks. See
        the function extended description for details.

    Returns
    -------
    root_address
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

    Keeps the communicator alive via state stored in the Distributed workers.

    Parameters
    ----------
    client
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


def get_shuffle_id() -> int:
    """Return the unique id for a new shuffle."""
    global _shuffle_counter  # noqa: PLW0603

    _shuffle_counter += 1
    return _shuffle_counter


def global_rmp_barrier(
    shuffle_ids: tuple[int, ...],
    dependencies: Sequence[Any],
):
    """
    Global barrier for rapidsmp shuffle.

    Notes
    -----
    A global barrier task does NOT need to be restricted
    to a specific Dask worker.
    """
    return shuffle_ids


def worker_rmp_barrier(
    shuffle_ids: tuple[int, ...],
    partition_count: int,
    global_barrier: tuple[int, ...],
):
    """
    Worker barrier for rapidsmp shuffle.

    Notes
    -----
    A worker barrier task DOES need to be restricted
    to a specific Dask worker.
    """
    for shuffle_id in shuffle_ids:
        shuffler = get_shuffler(shuffle_id)
        for pid in range(partition_count):
            shuffler.insert_finished(pid)
    return global_barrier


class DaskIntegration(Protocol):
    """dask-integration protocol."""

    @staticmethod
    def insert_partition(
        df: Any,
        on: Sequence[str],
        partition_count: int,
        shuffler: Shuffler,
    ) -> None:
        """
        Add a partition to a rapidsmp Shuffler.

        Parameters
        ----------
        df
            Partition data to add to a rapidsmp shuffler.
        on
            Sequence of column names to shuffle on.
        partition_count
            Number of output partitions for the current shuffle.
        shuffler
            The rapidsmp Shuffler object to extract from.
        """
        raise NotImplementedError("""Add a partition to a rapidsmp Shuffler.""")

    @staticmethod
    def extract_partition(
        partition_id: int,
        column_names: list[str],
        shuffler: Shuffler,
    ) -> Any:
        """
        Extract a partition from a rapidsmp Shuffler.

        Parameters
        ----------
        partition_id
            Partition id to extract.
        column_names
            Sequence of output column names.
        shuffler
            The rapidsmp Shuffler object to extract from.
        """
        raise NotImplementedError("""Extract a partition from a rapidsmp Shuffler.""")


def _insert_partition(
    df: Any,
    on: Sequence[str],
    partition_count: int,
    shuffle_id: int,
    *,
    callback: Callable | None = None,
):
    """Add a partition to a rapidsmp Shuffler."""
    if callback is None:
        raise ValueError("callback missing in _insert_partition.")
    return callback(
        df,
        on,
        partition_count,
        get_shuffler(shuffle_id),
    )


def _extract_partition(
    shuffle_id: int,
    partition_id: int,
    column_names: list[str],
    worker_barrier: tuple[int, ...],
    *,
    callback: Callable | None = None,
):
    """Extract a partition from a rapidsmp Shuffler."""
    if callback is None:
        raise ValueError("Missing callback in _extract_partition.")
    return callback(
        partition_id,
        column_names,
        get_shuffler(shuffle_id),
    )


def rapidsmp_shuffle_graph(
    input_name: str,
    output_name: str,
    column_names: Sequence[str],
    shuffle_on: Sequence[str],
    partition_count_in: int,
    partition_count_out: int,
    integration: DaskIntegration,
) -> MutableMapping[Any, Any]:
    """
    Return the task graph for a rapidsmp shuffle.

    Parameters
    ----------
    input_name
        The task name for input DataFrame tasks.
    output_name
        The task name for output DataFrame tasks.
    column_names
        Sequence of output column names.
    shuffle_on
        Sequence of column names to shuffle on (by hash).
    partition_count_in
        Partition count of input collection.
    partition_count_out
        Partition count of output collection.
    integration
        Dask-integration specification.
    """
    # Get the shuffle id
    client = get_dask_client()
    shuffle_id = get_shuffle_id()

    # Extract mapping between ranks and worker addresses
    worker_ranks: dict[int, str] = {
        v: k for k, v in client.run(get_worker_rank).items()
    }
    n_workers = len(worker_ranks)
    restricted_keys: MutableMapping[Any, str] = {}

    # Define task names for each stage of the shuffle
    insert_name = f"rmp-insert-{output_name}"
    global_barrier_1_name = f"rmp-global-barrier-1-{output_name}"
    global_barrier_2_name = f"rmp-global-barrier-2-{output_name}"
    worker_barrier_name = f"rmp-worker-barrier-{output_name}"

    # Stage a shuffler on every worker for this shuffle id
    client.run(
        get_shuffler,
        shuffle_id=shuffle_id,
        partition_count=partition_count_out,
    )

    # Add operation to submit each partition to the shuffler
    insert_partition = partial(
        _insert_partition,
        callback=integration.insert_partition,
    )
    graph: MutableMapping[Any, Any] = {
        (insert_name, pid): (
            insert_partition,
            (input_name, pid),
            shuffle_on,
            partition_count_out,
            shuffle_id,
        )
        for pid in range(partition_count_in)
    }

    # Add global barrier task
    graph[(global_barrier_1_name, 0)] = (
        global_rmp_barrier,
        (shuffle_id,),
        list(graph.keys()),
    )

    # Add worker barrier tasks
    worker_barriers: MutableMapping[Any, Any] = {}
    for rank, addr in worker_ranks.items():
        key = (worker_barrier_name, rank)
        worker_barriers[rank] = key
        graph[key] = (
            worker_rmp_barrier,
            (shuffle_id,),
            partition_count_out,
            (global_barrier_1_name, 0),
        )
        restricted_keys[key] = addr

    # Add global barrier task
    graph[(global_barrier_2_name, 0)] = (
        global_rmp_barrier,
        (shuffle_id,),
        list(worker_barriers.values()),
    )

    # Add extraction tasks
    output_keys = []
    extract_partition = partial(
        _extract_partition,
        callback=integration.extract_partition,
    )
    for part_id in range(partition_count_out):
        rank = part_id % n_workers
        output_keys.append((output_name, part_id))
        graph[output_keys[-1]] = (
            extract_partition,
            shuffle_id,
            part_id,
            column_names,
            (global_barrier_2_name, 0),
        )
        # Assume round-robin partition assignment
        restricted_keys[output_keys[-1]] = worker_ranks[rank]

    # Tell the scheduler to restrict the shuffle keys
    # to specific workers
    client._send_to_scheduler(
        {
            "op": "rmp_add_restricted_tasks",
            "tasks": restricted_keys,
        }
    )

    # Return the graph
    return graph


def rmp_worker_setup(
    dask_worker,
    *,
    pool_size: float = 0.75,
    spill_device: float = 0.0625,
):
    """
    Attach rapidsmp shuffling attributes to a Dask worker.

    Parameters
    ----------
    dask_worker
        The current Dask worker.
    pool_size
        The desired RMM pool size.
    spill_device
        GPU memory limit for shuffling.

    See Also
    --------
    bootstrap_dask_cluster
    """
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


def bootstrap_dask_cluster(client: Client):
    """
    Setup a Dask cluster for rapidsmp shuffling.

    Notes
    -----
    This utility must be executed before rapidsmp shuffling
    can be used within a Dask cluster. This function is called
    automatically by `rapidsmp.integrations.dask.get_client`.

    The `LocalRMPCluster` API is strongly recommended for
    local Dask-cluster generation, because it will automatically
    load the required `RMPSchedulerPlugin` (which must be loaded
    at cluster-initialization time).

    Parameters
    ----------
    client
        The current Dask client.

    See Also
    --------
    LocalRMPCluster
    """

    def rmp_plugin_not_registered(dask_scheduler=None) -> bool:
        for plugin in dask_scheduler.plugins:
            if "RMPSchedulerPlugin" in plugin:
                return False
        return True

    if client.id not in _initialized_clusters:
        # Check that RMPSchedulerPlugin is registered
        if client.run_on_scheduler(rmp_plugin_not_registered):
            raise ValueError("RMPSchedulerPlugin was not found on the scheduler.")

        # Setup "root" ucxx-comm rank
        workers = list(client.scheduler_info()["workers"])
        root_rank = [workers[0]]
        root_address_str = client.submit(
            rapidsmp_ucxx_rank_setup,
            nranks=len(workers),
            root_address_str=None,
            workers=root_rank,
            pure=False,
        ).result()

        # Setup other ucxx ranks
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

        # Setup the rapidsmp worker
        # TODO: Pass along non-default spilling options
        client.run(rmp_worker_setup)

        # Only run the above steps once
        _initialized_clusters.add(client.id)


class RMPSchedulerPlugin(SchedulerPlugin):
    """
    RAPIDS-MP Scheduler Plugin.

    The plugin helps manage integration with the RAPIDS-MP
    shuffle service by making it possible for the client
    to inform the scheduler of tasks that must be
    constrained to specific workers.

    See Also
    --------
    LocalRMPCluster
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


def get_dask_client() -> Client:
    """Get the current Dask client."""
    client = get_client()

    # Make sure the cluster supports rapidsmp
    bootstrap_dask_cluster(client)

    return client


def get_comm(dask_worker: Worker | None = None):
    """Get the RAPIDS-MP UCXX comm for a Dask worker."""
    dask_worker = dask_worker or get_worker()
    return dask_worker._rapidsmp_comm


def get_worker_rank(dask_worker: Worker | None = None):
    """Get the UCXX-comm rank for a Dask worker."""
    dask_worker = dask_worker or get_worker()
    return get_comm(dask_worker).rank


def get_shuffler(
    shuffle_id: int,
    partition_count: int | None = None,
    dask_worker: Worker | None = None,
):
    """
    Return the appropriate `Shuffler` object.

    Notes
    -----
    Whenever a new `Shuffler` object is created, it is
    cached as `dask_worker._rmp_shufflers[shuffle_id]`.

    Parameters
    ----------
    shuffle_id
        Unique ID for the shuffle operation.
    partition_count
        Output partition count for the shuffle operation.
    dask_worker
        The current dask worker.

    Returns
    -------
    The active rapidsmp `Shuffler` object associated with
    the specified `shuffle_id`, `partition_count` and
    `dask_worker`.
    """
    dask_worker = dask_worker or get_worker()

    if shuffle_id not in dask_worker._rmp_shufflers:
        if partition_count is None:
            raise ValueError(
                "Need partition_count to create new shuffler."
                f" shuffle_id: {shuffle_id}\n"
                f" Shufflers: {dask_worker._rmp_shufflers}"
            )
        dask_worker._rmp_shufflers[shuffle_id] = Shuffler(
            get_comm(dask_worker),
            op_id=shuffle_id,
            total_num_partitions=partition_count,
            stream=DEFAULT_STREAM,
            br=get_buffer_resource(dask_worker),
        )

    return dask_worker._rmp_shufflers[shuffle_id]


def get_buffer_resource(dask_worker: Worker | None = None):
    """Get the RAPIDS-MP buffer resource for a Dask worker."""
    dask_worker = dask_worker or get_worker()
    return dask_worker._buffer_resource


def get_memory_resource(dask_worker: Worker | None = None):
    """Get the RMM memory resource for a Dask worker."""
    dask_worker = dask_worker or get_worker()
    return dask_worker._memory_resource


class LocalRMPCluster(LocalCUDACluster):
    """
    Local rapidsmp Dask cluster.

    Notes
    -----
    This class wraps `dask_cuda.LocalCUDACluster`, and
    automatically registers an `RMPSchedulerPlugin` at
    initialization time. This plugin allows a distributed
    client to inform the scheduler of specific worker
    restrictions at graph-construction time. This feature
    is currently needed for dask + rapidsmp integration.
    """

    def __init__(self, **kwargs):
        self._rmp_shuffle_counter = 0
        preloads = config.get("distributed.scheduler.preload")
        preloads.append("rapidsmp.examples.dask")
        config.set({"distributed.scheduler.preload": preloads})
        super().__init__(**kwargs)


def dask_setup(scheduler):
    """Setup dask cluster."""
    plugin = RMPSchedulerPlugin(scheduler)
    scheduler.add_plugin(plugin)
