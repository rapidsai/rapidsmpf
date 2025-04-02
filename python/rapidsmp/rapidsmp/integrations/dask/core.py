# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffler integration for Dask Distributed clusters."""

from __future__ import annotations

import logging
import threading
import weakref
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast, runtime_checkable

import ucxx._lib.libucxx as ucx_api
from distributed import get_client, get_worker, wait
from distributed.diagnostics.plugin import SchedulerPlugin

import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmp.buffer.buffer import MemoryType
from rapidsmp.buffer.resource import BufferResource, LimitAvailableMemory
from rapidsmp.communicator.communicator import Communicator
from rapidsmp.communicator.ucxx import barrier, get_root_ucxx_address, new_communicator
from rapidsmp.integrations.dask.shuffler import get_shuffle_id
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
        dask_worker._rmp_buffer_resource = BufferResource(mr, memory_available)


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
