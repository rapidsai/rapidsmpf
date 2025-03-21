# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Integration for Dask Distributed clusters."""

from __future__ import annotations

import asyncio
import logging
import threading
import weakref
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast, runtime_checkable

import dask.utils
import ucxx._lib.libucxx as ucx_api
from distributed import get_client, get_worker
from distributed.diagnostics.plugin import SchedulerPlugin, WorkerPlugin
from distributed.utils import Deadline

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


_dask_logger = logging.getLogger("distributed.worker")
DataFrameT = TypeVar("DataFrameT")


# Worker and Client caching utilities
_worker_thread_lock: threading.RLock = threading.RLock()
_initialized_clusters: set[str] = set()
_shuffle_counter: int = 0


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

    This function is meant to be a no-op.
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
    """
    if callback is None:
        raise ValueError("callback missing in _insert_partition.")
    callback(
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
    A new :class:`Shuffler` object must be staged on every worker
    in the current Dask cluster.

    **Insertion phase**
    Each input partition is split into a dictionary of chunks,
    and that dictionary is passed to the appropriate :class:`Shuffler`
    object (using `Shuffler.insert_chunks`).

    The insertion phase will include a single task for each of
    the ``partition_count_in`` partitions in the input DataFrame.
    The partitioning and insertion logic must be defined by the
    ``insert_partition`` classmethod of the ``integration`` argument.

    Insertion tasks are NOT restricted to specific Dask workers.
    These tasks may run anywhere in the cluster.

    **Barrier phase**
    All :class:`Shuffler` objects must be 'informed' that the insertion
    phase is complete (on all workers) before the subsequent
    extraction phase begins. We call this synchronization step
    the 'barrier phase'.

    The barrier phase comprises three types of barrier tasks:

    1. First global barrier - A single barrier task is used to
    signal that all input partitions have been submitted to
    a :class:`Shuffler` object on one of the workers. This task may
    also run anywhere on the cluster, but it must depend on
    ALL insertion tasks.

    2. Worker barrier(s) - Each worker must execute a single
    worker-barrier task. This task will call `insert_finished`
    for every output partition on the local :class:`Shuffler`. These
    tasks must be restricted to specific workers, and they
    must all depend on the first global barrier.

    3. Second global barrier - A single barrier task is used
    to signal that all workers are ready to begin the
    extraction pahse. This task may run anywhere on the cluster,
    but it must depend on all worker-barrier tasks.

    **Extraction phase**
    Each output partition is extracted from the local
    :class:`Shuffler` object on the worker (using `Shuffler.wait_on`
    and `rapidsmp.shuffler.unpack_and_concat`).

    The extraction phase will include a single task for each of
    the ``partition_count_out`` partitions in the shuffled output
    DataFrame. The extraction logic must be defined by the
    ``extract_partition`` classmethod of the ``integration`` argument.

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

    # Add tasks to insert each partition into the shuffler
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

    return graph


def rmp_worker_setup(
    dask_worker: Worker,
    *,
    pool_size: float = 0.75,
    spill_device: float = 0.50,
    enable_statistics: bool = True,
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
    with _worker_thread_lock:
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


async def bootstrap_dask_cluster_async(
    client: Client,
    *,
    pool_size: float = 0.75,
    spill_device: float = 0.50,
    enable_statistics: bool = True,
) -> None:
    """
    Setup an asynchronous Dask cluster for rapidsmp shuffling.

    Parameters
    ----------
    client
        The current Dask client.
    pool_size
        The desired RMM pool size.
    spill_device
        GPU memory limit for shuffling.
    enable_statistics
        Whether to track shuffler statistics.

    See Also
    --------
    bootstrap_dask_cluster
        Setup a synchronous Dask cluster for rapidsmp shuffling.

    Notes
    -----
    This utility must be executed before rapidsmp shuffling
    can be used within a Dask cluster. This function is called
    automatically by `rapidsmp.integrations.dask.get_client`.
    """
    if not client.asynchronous:
        raise ValueError("Client must be asynchronous")

    if client.id in _initialized_clusters:
        return

    # Bootstrap the scheduler.
    scheduler_plugin = RMPSchedulerPlugin()
    await client.register_plugin(scheduler_plugin)

    # Bootstrap the workers.
    worker_plugin = RMPWorkerPlugin(
        worker_addresses=sorted(client.scheduler_info()["workers"]),
        pool_size=pool_size,
        spill_device=spill_device,
        enable_statistics=enable_statistics,
    )
    await client.register_plugin(worker_plugin)
    _initialized_clusters.add(client.id)


def bootstrap_dask_cluster(
    client: Client,
    *,
    pool_size: float = 0.75,
    spill_device: float = 0.50,
    enable_statistics: bool = True,
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
    enable_statistics
        Whether to track shuffler statistics.

    See Also
    --------
    bootstrap_dask_cluster_async
        Setup an asynchronous Dask cluster for rapidsmp shuffling.

    Notes
    -----
    This utility must be executed before rapidsmp shuffling
    can be used within a Dask cluster. This function is called
    automatically by `rapidsmp.integrations.dask.get_client`.
    """
    if client.asynchronous:
        raise ValueError("Client must be synchronous")

    if client.id in _initialized_clusters:
        return

    # Scheduler stuff
    scheduler_plugin = RMPSchedulerPlugin()
    client.register_plugin(scheduler_plugin)

    # Worker stuff
    worker_plugin = RMPWorkerPlugin(
        worker_addresses=sorted(client.scheduler_info()["workers"]),
        pool_size=pool_size,
        spill_device=spill_device,
        enable_statistics=enable_statistics,
    )
    client.register_plugin(worker_plugin)
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


class RMPWorkerPlugin(WorkerPlugin):
    """
    RAPIDS-MP Worker Plugin.

    This plugin ensures that the cluster is ready to shuffle data by
    establishing a UCXX communicators between all workers.

    Parameters
    ----------
    worker_addresses : list[str]
        The addresses of each worker in the cluster. Use
        :meth:`distributed.Client.scheduler_info()["workers"]` to get this list.
    pool_size : float
        The desired RMM pool size.
    spill_device : float
        GPU memory limit for shuffling.
    enable_statistics : bool
        Whether to track shuffler statistics.
    ready_timeout : float | str
        Timeout for a worker to become ready. For the root worker, this includes
        the time to create its own UCXX communicator and wait for all other
        nodes to check in. For non-root workers, this includes the time for it
        to get the UCXX address of the root worker. Can be a
        float (interpreted as seconds) or a string with a unit parsed by
        `dask.utils.parse_timedelta`.

        This defaults to ``distributed.comm.timeouts.connect`` from the
        Dask configuration.
    """

    idempotent: bool = True

    def __init__(
        self,
        *,
        worker_addresses: Sequence[str],
        pool_size: float = 0.75,
        spill_device: float = 0.50,
        enable_statistics: bool = True,
        ready_timeout: float | str | None = None,
    ) -> None:
        self.worker_addresses = worker_addresses
        self.pool_size = pool_size
        self.spill_device = spill_device
        self.enable_statistics = enable_statistics
        self.ready_timeout = dask.utils.parse_timedelta(
            ready_timeout or dask.config.get("distributed.comm.timeouts.connect", 30)
        )
        self.worker = None
        self._heard_from: set[str] = set()
        self._ready = asyncio.Event()
        if len(self.worker_addresses) == 1:
            # Avoid a deadlock later on. The root node will wait for some
            # other node to call its ``_root_ucxx_address`` handler. If there
            # are no other nodes, nobody calls, and we wait forever.
            self._ready.set()

    @property
    def _is_root(self) -> bool:
        """
        Whether this worker is the root node (first in the list).

        Returns
        -------
        bool
            True when this worker is the root node.
        """
        assert self.worker is not None
        return self.worker.address == self.worker_addresses[0]

    async def setup(self, worker: Worker) -> None:
        """
        Set up the worker for rapidsmp shuffling.

        This will set up the rapidsmp resources on the worker, including

        - A UCXX communicator between all workers in the Dask cluster, with the
          first worker being the root node.
        - Various buffer resources for shuffling and spilling.
        - A Statistics object for tracking shuffle performance.

        Parameters
        ----------
        worker : distributed.Worker
            The Dask worker this setup is being called on. This will be supplied
            by the distributed runtime.
        """
        # Bootstrapping algorithm:
        #
        # We need to stand up a UCXX communicator between all workers. The only
        # real trick is figuring out the **UCXX** address of the root node. We
        # don't know that address before we get here (because we haven't set up
        # UCXX yet), so we need to figure that out as we go. Two facts let us
        # figure that out:
        #
        # 1. The Dask comms system is already up and running
        # 2. We know the **Dask** addresses of each node in the cluster
        #
        # So we can go through the normal dance of checking if we're the root
        # node (i.e. we're the first worker in the list). If we are, we'll
        # create the UCXX comm, which gets set on the Dask Worker object.
        #
        # Non-root nodes will need to ask the root node for it's UCXX address.
        # It knows the Dask address of the root node (again, first one in the
        # list) and can use that to make a Dask RPC call to the root node. The
        # root node replies with it's UCXX address, and then the worker can go
        # on its way.
        #
        # There's a couple of wrinkles here that make the implementation more
        # complicated. This whole things ends with a UCXX barrier. We don't want
        # to return flow to the user until that barrier has been passed. But if
        # the root node waits at that barrier the cluster will deadlock, because
        # no one is there to answer the RPC call (apparently these happen on the
        # same thread?). So we use a little asyncio.Event to track whether the
        # root node has heard from the workers before advancing to the barrier.
        #
        # The upshot of all this complexity is that we don't have to do a
        # two-stage bootstrapping, and we don't have to mess with our futures
        # showing up in the user's Dask dashboard.
        self.worker = worker

        if hasattr(worker, "_rapidsmp_comm"):
            # We've already been called.
            return

        worker.handlers["_root_ucxx_address"] = self._root_ucxx_address
        nranks = len(self.worker_addresses)
        root_dask_address = self.worker_addresses[0]
        _dask_logger.info("RMPWorkerPlugin setup. address=%s", worker.address)

        if self._is_root:
            _dask_logger.debug(
                "RMPWorkerPlugin setup. address=%s stage=create-communicator-start",
                worker.address,
            )
            comm = new_communicator(nranks, None, None)
            _dask_logger.debug(
                "RMPWorkerPlugin setup. address=%s rank=%d stage=create-communicator-done",
                worker.address,
                comm.rank,
            )
            worker._rapidsmp_comm = comm
            comm.logger.trace(f"Rank {comm.rank} created")
            # We're ready once we've heard from every other node. The handler
            # will take care of setting the event once it's heard from everyone.
            try:
                await asyncio.wait_for(self._ready.wait(), timeout=self.ready_timeout)
            except asyncio.TimeoutError as e:
                missing = set(self.worker_addresses) - self._heard_from
                missing.discard(worker.address)
                msg = f"Timeout waiting for workers to bootstrap. Root = {worker.address}, missing = {sorted(missing)}"
                raise RuntimeError(msg) from e

        else:
            # Non-root nodes ask the root node for its ucxx address.
            root_ucxx_address = None
            attempt = 0
            deadline = Deadline.after(self.ready_timeout)

            while not deadline.expired:
                _dask_logger.debug(
                    "RMPWorkerPlugin setup. address=%s stage=get-root-ucxx-address",
                    worker.address,
                )
                root_ucxx_address = await worker.rpc(
                    self.worker_addresses[0]
                )._root_ucxx_address(caller=worker.address)
                if root_ucxx_address is not None:
                    _dask_logger.debug(
                        "RMPWorkerPlugin setup. address=%s rank=%d stage=got-root-ucxx-address",
                        worker.address,
                        root_dask_address,
                    )
                    break
                else:
                    _dask_logger.debug(
                        "RMPWorkerPlugin setup. address=%s attempt=%d stage=get-root-ucxx-address-retry",
                        worker.address,
                        attempt,
                    )
                    await asyncio.sleep(
                        _exponential_backoff(
                            attempt, multiplier=1, exponential_base=0.5, max_interval=10
                        )
                    )
                    attempt += 1

            if root_ucxx_address is None:
                _dask_logger.warning(
                    "RMPWorkerPlugin setup. address=%s stage=get-root-ucxx-address-failed",
                    worker.address,
                )
                raise RuntimeError(
                    f"Worker {worker.address} failed to get root ucxx address from {self.worker_addresses[0]}"
                )

            root_address = ucx_api.UCXAddress.create_from_buffer(root_ucxx_address)
            comm = new_communicator(nranks, None, root_address)
            worker._rapidsmp_comm = comm

            comm.logger.trace(f"Rank {comm.rank} created")

        # Now we wait for everyone else to be ready.
        _dask_logger.debug(
            "RMPWorkerPlugin setup. address=%s stage=barrier-wait", worker.address
        )
        comm.logger.trace(f"Rank {comm.rank} setup barrier")
        barrier(comm)
        comm.logger.trace(f"Rank {comm.rank} setup barrier passed")
        _dask_logger.debug(
            "RMPWorkerPlugin setup. address=%s stage=barrier-passed", worker.address
        )

        await asyncio.to_thread(
            rmp_worker_setup,
            worker,
            pool_size=self.pool_size,
            spill_device=self.spill_device,
            enable_statistics=self.enable_statistics,
        )
        _dask_logger.debug(
            "RMPWorkerPlugin setup. address=%s stage=finished", worker.address
        )

        return None

    def _root_ucxx_address(self, *, caller: str) -> bytes | None:
        """
        Get the UCXX address of the root worker.

        This is registered as a handler on the Dask Workers. Each non-root
        worker is expected to call the root worker using this handler.

        Parameters
        ----------
        caller : str
            The Dask address of the caller. This is used by handler to track
            which workers have made it to this point.

        Returns
        -------
        bytes, optional
            The UCXX address of the root worker.

        Notes
        -----
        The root worker uses the ``caller`` addresses along with the
        `asyncio.Event` on the plugin class to know when it's safe to proceed to
        the barrier (which blocks).
        """
        rapidsmp_comm = getattr(self.worker, "_rapidsmp_comm", None)
        if self._is_root and rapidsmp_comm is not None:
            address = get_root_ucxx_address(rapidsmp_comm)
            self._heard_from.add(caller)
            if len(self._heard_from) == len(self.worker_addresses) - 1:
                # -1, because we don't count ourselves
                self._ready.set()
            return address

        return None


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
    assert isinstance(dask_worker._rapidsmp_comm, Communicator), (
        f"Expected Communicator, got {dask_worker._rapidsmp_comm}"
    )
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
    Return the appropriate :class:`Shuffler` object.

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
    The active rapidsmp :class:`Shuffler` object associated with
    the specified ``shuffle_id``, ``partition_count`` and
    ``dask_worker``.

    Notes
    -----
    Whenever a new :class:`Shuffler` object is created, it is
    saved as ``dask_worker._rmp_shufflers[shuffle_id]``.

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


def _exponential_backoff(
    attempt: int, multiplier: float, exponential_base: float, max_interval: float
) -> float:
    """
    Calculate the duration of an exponential backoff.

    Parameters
    ----------
    attempt : int
        The attempt number. Increment this between attempts.
    multiplier : float
        The multiplier for the exponential backoff.
    exponential_base : float
        The base for the exponential backoff.
    max_interval : float
        The maximum interval for the exponential backoff.

    Returns
    -------
    float
        The duration of the exponential backoff.
    """
    try:
        interval = multiplier * exponential_base**attempt
    except OverflowError:
        return max_interval

    return min(max_interval, interval)
