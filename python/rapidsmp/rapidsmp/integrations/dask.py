# Copyright (c) 2025, NVIDIA CORPORATION.
"""Integration for Dask Distributed clusters."""

from __future__ import annotations

import asyncio
import threading
import weakref
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast, runtime_checkable

import ucxx._lib.libucxx as ucx_api
from dask import config
from dask_cuda import LocalCUDACluster
from distributed import get_client, get_worker, wait
from distributed.diagnostics.plugin import SchedulerPlugin

import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmp.buffer.buffer import MemoryType
from rapidsmp.buffer.resource import BufferResource, LimitAvailableMemory
from rapidsmp.communicator.communicator import Communicator
from rapidsmp.communicator.ucxx import barrier, get_root_ucxx_address, new_communicator
from rapidsmp.shuffler import Shuffler
from rapidsmp.statistics import Statistics

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping, Sequence

    from distributed import Client, Worker
    from distributed.scheduler import Scheduler, TaskState


DataFrameT = TypeVar("DataFrameT")


# Worker and Client caching utilities
_worker_thread_lock: threading.RLock = threading.RLock()
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

    comm: Communicator
    if root_address_str is None:
        comm = new_communicator(nranks, None, None)
        comm.logger.trace(f"Rank {comm.rank} created")
        dask_worker._rapidsmp_comm = comm
        return get_root_ucxx_address(comm)
    else:
        if hasattr(dask_worker, "_rapidsmp_comm"):
            assert isinstance(dask_worker._rapidsmp_comm, Communicator)
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


async def rapidsmp_ucxx_comm_setup(client: Client) -> None:
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
    """
    Return the unique id for a new shuffle.

    Returns
    -------
    The enumerated integer id for the current shuffle.
    """
    global _shuffle_counter  # noqa: PLW0603

    _shuffle_counter += 1
    return _shuffle_counter


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
    """


def worker_rmp_barrier(
    shuffle_ids: tuple[int, ...],
    partition_count: int,
    dependency: None,
) -> None:
    """
    Worker barrier for rapidsmp shuffle.

    Parameters
    ----------
    shuffle_ids
        Tuple of shuffle ids associated with the current
        task graph. This tuple will only contain a single
        integer when `rapidsmp_shuffle_graph` is used for
        graph generation.
    partition_count
        Number of output partitions for the current shuffle.
    dependency
        Null argument used to enforce barrier dependencies.

    Notes
    -----
    A worker barrier task DOES need to be restricted
    to a specific Dask worker.
    """
    for shuffle_id in shuffle_ids:
        shuffler = get_shuffler(shuffle_id)
        for pid in range(partition_count):
            shuffler.insert_finished(pid)


@runtime_checkable
class DaskIntegration(Protocol[DataFrameT]):
    """
    dask-integration protocol.

    This protocol can be used to implement a rapidsmp-shuffle
    operation using a Dask task graph.
    """

    @staticmethod
    def insert_partition(
        df: DataFrameT,
        on: Sequence[str],
        partition_count: int,
        shuffler: Shuffler,
    ) -> None:
        """
        Add a partition to a rapidsmp Shuffler.

        Parameters
        ----------
        df
            DataFrame partition to add to a rapidsmp shuffler.
        on
            Sequence of column names to shuffle on.
        partition_count
            Number of output partitions for the current shuffle.
        shuffler
            The rapidsmp Shuffler object to extract from.
        """

    @staticmethod
    def extract_partition(
        partition_id: int,
        column_names: list[str],
        shuffler: Shuffler,
    ) -> DataFrameT:
        """
        Extract a DataFrame partition from a rapidsmp Shuffler.

        Parameters
        ----------
        partition_id
            Partition id to extract.
        column_names
            Sequence of output column names.
        shuffler
            The rapidsmp Shuffler object to extract from.

        Returns
        -------
        A shuffled DataFrame partition.
        """


def _insert_partition(
    callback: Callable[[DataFrameT, Sequence[str], int, Shuffler], None],
    df: DataFrameT,
    on: Sequence[str],
    partition_count: int,
    shuffle_id: int,
) -> None:
    """
    Add a partition to a rapidsmp Shuffler.

    Parameters
    ----------
    callback
        Insertion callback function. This function must be
        the `insert_partition` attribute of a `DaskIntegration`
        protocol.
    df
        DataFrame partition to add to a rapidsmp shuffler.
    on
        Sequence of column names to shuffle on.
    partition_count
        Number of output partitions for the current shuffle.
    shuffle_id
        The rapidsmp shuffle id.

    Returns
    -------
    None
    """
    if callback is None:
        raise ValueError("callback missing in _insert_partition.")
    return callback(
        df,
        on,
        partition_count,
        get_shuffler(shuffle_id),
    )


def _extract_partition(
    callback: Callable[[int, Sequence[str], Shuffler], DataFrameT],
    shuffle_id: int,
    partition_id: int,
    column_names: list[str],
    worker_barrier: tuple[int, ...],
) -> DataFrameT:
    """
    Extract a partition from a rapidsmp Shuffler.

    Parameters
    ----------
    callback
        Insertion callback function. This function must be
        the `extract_partition` attribute of a `DaskIntegration`
        protocol.
    shuffle_id
        The rapidsmp shuffle id.
    partition_id
        Partition id to extract.
    column_names
        Sequence of output column names.
    worker_barrier
        Worker-barrier task dependency. This value should
        not be used for compute logic.

    Returns
    -------
    Extracted DataFrame partition.
    """
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
) -> dict[Any, Any]:
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

    Returns
    -------
    A valid task graph for Dask execution.

    Notes
    -----
    A rapidsmp shuffle operation comprises four general phases:

    **Staging phase**
    A new `Shuffler` object must be staged on every worker
    in the current Dask cluster.

    **Insertion phase**
    Each input partition is split into a dictionary of chunks,
    and that dictionary is passed to the appropriate `Shuffler`
    object (using `Shuffler.insert`).

    The insertion phase will include a single task for each of
    the `partition_count_in` partitions in the input DataFrame.
    The partitioning and insertion logic must be defined by the
    `insert_partition` classmethod of the `integration` argument.

    Insertion tasks are NOT restricted to specific Dask workers.
    These tasks may run anywhere in the cluster.

    **Barrier phase**
    All `Shuffler` objects must be 'informed' that the insertion
    phase is complete (on all workers) before the subsequent
    extraction phase begins. We call this synchronization step
    the 'barrier phase'.

    The barrier phase comprises three types of barrier tasks:

    1. First global barrier - A single barrier task is used to
    signal that all input partitions have been submitted to
    a `Shuffler` object on one of the workers. This task may
    also run anywhere on the cluster, but it must depend on
    ALL insertion tasks.

    2. Worker barrier(s) - Each worker must execute a single
    worker-barrier task. This task will call `insert_finished`
    for every output partition on the local `Shuffler`. These
    tasks must be restricted to specific workers, and they
    must all depend on the first global barrier.

    3. Second global barrier - A single barrier task is used
    to signal that all workers are ready to begin the
    extraction pahse. This task may run anywhere on the cluster,
    but it must depend on all worker-barrier tasks.

    **Extraction phase**
    Each output partition is extracted from the local
    `Shuffler` object on the worker (using `Shuffler.wait_on`
    and `rapidsmp.shuffler.unpack_and_concat`).

    The extraction phase will include a single task for each of
    the `partition_count_out` partitions in the shuffled output
    DataFrame. The extraction logic must be defined by the
    `extract_partition` classmethod of the `integration` argument.

    Extraction tasks must be restricted to specific Dask workers,
    and they must also depend on the second global-barrier task.
    """
    # Get the shuffle id
    client = get_dask_client()
    shuffle_id = get_shuffle_id()

    # Check integration argument
    if not isinstance(integration, DaskIntegration):
        raise TypeError(f"Expected DaskIntegration object, got {integration}.")

    # Extract mapping between ranks and worker addresses
    worker_ranks: dict[int, str] = {
        v: k for k, v in client.run(get_worker_rank).items()
    }
    n_workers = len(worker_ranks)
    restricted_keys: MutableMapping[Any, str] = {}

    # Define task names for each phase of the shuffle
    insert_name = f"rmp-insert-{output_name}"
    global_barrier_1_name = f"rmp-global-barrier-1-{output_name}"
    global_barrier_2_name = f"rmp-global-barrier-2-{output_name}"
    worker_barrier_name = f"rmp-worker-barrier-{output_name}"

    # Stage a shuffler on every worker for this shuffle id
    client.run(
        _stage_shuffler,
        shuffle_id=shuffle_id,
        partition_count=partition_count_out,
    )

    # Make sure shuffle_on does not contain duplicate keys
    if len(set(shuffle_on)) != len(shuffle_on):
        raise ValueError(f"Got duplicate keys in shuffle_on: {shuffle_on}")

    # Add operation to submit each partition to the shuffler
    graph: dict[Any, Any] = {
        (insert_name, pid): (
            _insert_partition,
            integration.insert_partition,
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
        list(graph.keys()),
    )

    # Add worker barrier tasks
    worker_barriers: dict[Any, Any] = {}
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
        list(worker_barriers.values()),
    )

    # Add extraction tasks
    output_keys = []
    for part_id in range(partition_count_out):
        rank = part_id % n_workers
        output_keys.append((output_name, part_id))
        graph[output_keys[-1]] = (
            _extract_partition,
            integration.extract_partition,
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
    dask_worker: Worker,
    *,
    pool_size: float = 0.75,
    spill_device: float = 0.50,
) -> None:
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
        Setup a Dask cluster for rapidsmp shuffling.

    Notes
    -----
    This function is expected to run on a Dask worker.
    """
    with _worker_thread_lock:
        if hasattr(dask_worker, "_rmp_shufflers"):
            return  # Worker already initialized

        # Add empty list of active shufflers
        dask_worker._rmp_shufflers = {}

        # Print statstics at worker shutdown
        dask_worker._rmp_statistics = Statistics(enable=True)
        weakref.finalize(
            dask_worker,
            lambda name, stats: print(name, stats.report()),
            name=str(dask_worker),
            stats=dask_worker._rmp_statistics,
        )

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
        memory_available = {
            MemoryType.DEVICE: LimitAvailableMemory(
                mr, limit=int(available_memory * spill_device)
            )
        }
        dask_worker._rmp_buffer_resource = BufferResource(mr, memory_available)


def bootstrap_dask_cluster(
    client: Client,
    *,
    pool_size: float = 0.75,
    spill_device: float = 0.50,
) -> None:
    """
    Setup a Dask cluster for rapidsmp shuffling.

    Parameters
    ----------
    client
        The current Dask client.
    pool_size
        The desired RMM pool size.
    spill_device
        GPU memory limit for shuffling.

    See Also
    --------
    LocalRMPCluster
        Local rapidsmp-specific Dask cluster.

    Notes
    -----
    This utility must be executed before rapidsmp shuffling
    can be used within a Dask cluster. This function is called
    automatically by `rapidsmp.integrations.dask.get_client`.

    The `LocalRMPCluster` API is strongly recommended for
    local Dask-cluster generation, because it will automatically
    load the required `RMPSchedulerPlugin` (which must be loaded
    at cluster-initialization time).
    """

    def rmp_plugin_not_registered(dask_scheduler: Scheduler | None = None) -> bool:
        """
        Check if a RMPSchedulerPlugin is registered.

        Parameters
        ----------
        dask_scheduler
            Dask scheduler object.

        Returns
        -------
        Whether a RMPSchedulerPlugin is registered.
        """
        assert dask_scheduler is not None
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
        client.run(
            rmp_worker_setup,
            pool_size=pool_size,
            spill_device=spill_device,
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

    Parameters
    ----------
    scheduler
        The current Dask scheduler object.

    See Also
    --------
    LocalRMPCluster
        Local rapidsmp-specific Dask cluster.
    """

    scheduler: Scheduler
    _rmp_restricted_tasks: dict[str, str]

    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
        self.scheduler.stream_handlers.update(
            {"rmp_add_restricted_tasks": self.rmp_add_restricted_tasks}
        )
        self.scheduler.add_plugin(self, name="rampidsmp_manger")
        self._rmp_restricted_tasks = {}

    def rmp_add_restricted_tasks(self, *args: Any, **kwargs: Any) -> None:
        """
        Add restricted tasks.

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
        Update graph on scheduler.

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

    # Make sure the cluster supports rapidsmp
    bootstrap_dask_cluster(client)

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
    assert isinstance(dask_worker._rapidsmp_comm, Communicator)
    return dask_worker._rapidsmp_comm


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


def get_shuffler(
    shuffle_id: int,
    partition_count: int | None = None,
    dask_worker: Worker | None = None,
) -> Shuffler:
    """
    Return the appropriate `Shuffler` object.

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

    Notes
    -----
    Whenever a new `Shuffler` object is created, it is
    cached as `dask_worker._rmp_shufflers[shuffle_id]`.

    This function is expected to run on a Dask worker.
    """
    dask_worker = dask_worker or get_worker()
    with _worker_thread_lock:
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
                br=dask_worker._rmp_buffer_resource,
                statistics=dask_worker._rmp_statistics,
            )
    return cast(Shuffler, dask_worker._rmp_shufflers[shuffle_id])


def _stage_shuffler(
    shuffle_id: int,
    partition_count: int,
    dask_worker: Worker | None = None,
) -> None:
    """
    Stage a shuffler object without returning it.

    Parameters
    ----------
    shuffle_id
        Unique ID for the shuffle operation.
    partition_count
        Output partition count for the shuffle operation.
    dask_worker
        The current dask worker.

    Notes
    -----
    This function is expected to run on a Dask worker.
    """
    dask_worker = dask_worker or get_worker()
    get_shuffler(
        shuffle_id,
        partition_count=partition_count,
        dask_worker=dask_worker,
    )


class LocalRMPCluster(LocalCUDACluster):
    """
    Local rapidsmp Dask cluster.

    Parameters
    ----------
    **kwargs
        Key-word arguments to be passed through to
        `dask_cuda.LocalCUDACluster`.

    Notes
    -----
    This class wraps `dask_cuda.LocalCUDACluster`, and
    automatically registers an `RMPSchedulerPlugin` at
    initialization time. This plugin allows a distributed
    client to inform the scheduler of specific worker
    restrictions at graph-construction time. This feature
    is currently needed for dask + rapidsmp integration.
    """

    def __init__(self, **kwargs: Any):
        self._rmp_shuffle_counter = 0
        preloads = config.get("distributed.scheduler.preload")
        preloads.append("rapidsmp.integrations.dask")
        config.set({"distributed.scheduler.preload": preloads})
        super().__init__(**kwargs)


def dask_setup(scheduler: Scheduler) -> None:
    """
    Setup dask cluster.

    Parameters
    ----------
    scheduler
        Dask scheduler object.
    """
    plugin = RMPSchedulerPlugin(scheduler)
    scheduler.add_plugin(plugin)
