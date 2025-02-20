# Copyright (c) 2025, NVIDIA CORPORATION.
"""Dask integration example."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dask.dataframe as dd
import pandas as pd
from dask import config
from dask.tokenize import tokenize
from dask_cuda import LocalCUDACluster
from distributed import get_client, get_worker
from distributed.diagnostics.plugin import SchedulerPlugin

import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmp.buffer.buffer import MemoryType
from rapidsmp.buffer.resource import BufferResource, LimitAvailableMemory
from rapidsmp.integrations.dask import rapidsmp_ucxx_comm_setup_sync
from rapidsmp.shuffler import Shuffler, partition_and_pack, unpack_and_concat
from rapidsmp.testing import pylibcudf_to_cudf_dataframe

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from distributed.scheduler import Scheduler, TaskState
    from distributed.worker import Worker

    import cudf


_initialized_clusters = set()


def rmp_setup(dask_worker, pool_size: float = 0.8, spill_device: float = 0.4):
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
    dask_worker._buffer_resource = BufferResource(mr, memory_available)


def initialize_ucxx_comms(client):
    """Initialize RAPIDS-MP UCXX comms for shuffling."""
    token = tokenize(client.id, list(client.scheduler_info()["workers"].keys()))
    if token not in _initialized_clusters:
        rapidsmp_ucxx_comm_setup_sync(client)
        client.run(rmp_setup)
        _initialized_clusters.add(token)


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


def get_buffer_resource(worker: Worker | None = None):
    """Get the RAPIDS-MP buffer resource for this worker."""
    worker = worker or get_worker()
    return worker._buffer_resource


def get_shuffler(shuffle_id: int, partition_count: int | None = None):
    """Get the active RAPIDS-MP shuffler."""
    worker = get_worker()

    if shuffle_id not in worker._rmp_shufflers:
        if partition_count is None:
            raise RuntimeError(
                "Need partition_count to create new shuffler."
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


def rmp_shuffle_insert(
    df: cudf.DataFrame,
    on: Sequence[str],
    partition_count: int,
    shuffle_id: int,
):
    """Add chunks to an RMP shuffler."""
    shuffler = get_shuffler(shuffle_id, partition_count=partition_count)

    columns_to_hash = tuple(list(df.columns).index(val) for val in on)
    packed_inputs = partition_and_pack(
        df.to_pylibcudf()[0],
        columns_to_hash=columns_to_hash,
        num_partitions=partition_count,
        stream=DEFAULT_STREAM,
        device_mr=rmm.mr.get_current_device_resource(),
    )
    shuffler.insert_chunks(packed_inputs)

    # Pass back a pd.DataFrame object to make this
    # a proper Dask-DataFrame collection (for now)
    return pd.DataFrame({"id": [shuffle_id]})


def rmp_shuffle_extract(
    shuffle_id: int,
    partition_id: int,
    column_names: list[str],
    barrier_dep: pd.DataFrame,
):
    """Extract a finished partition from the RMP shuffler."""
    shuffler = get_shuffler(shuffle_id)
    shuffler.wait_on(partition_id)
    table = unpack_and_concat(
        shuffler.extract(partition_id),
        stream=DEFAULT_STREAM,
        device_mr=rmm.mr.get_current_device_resource(),
    )
    return pylibcudf_to_cudf_dataframe(
        table,
        column_names=column_names,
    )


def get_worker_rank(dask_worker: Worker):
    """Get the UCXX-comm rank for this worker."""
    return get_comm(dask_worker).rank


def global_rmp_barrier(
    shuffle_id: int, partition_count: int, insert_results: Sequence[pd.DataFrame]
):
    """Global RMP shuffle barrier."""
    assert len(insert_results) == partition_count
    return shuffle_id


def worker_rmp_barrier(
    shuffle_id: int,
    partition_count: int,
    global_barrier: int,
):
    """Worker-specific RMP shuffle barrier."""
    shuffler = get_shuffler(shuffle_id)
    for pid in range(partition_count):
        shuffler.insert_finished(pid)
    return global_barrier


class LocalRMPCluster(LocalCUDACluster):
    """Local RAPIDSMP Dask cluster."""

    def __init__(self, **kwargs):
        self._rmp_shuffle_counter = 0
        preloads = config.get("distributed.scheduler.preload")
        preloads.append("rapidsmp.examples.dask_shuffle")
        config.set({"distributed.scheduler.preload": preloads})
        super().__init__(**kwargs)

    def get_shuffle_id(self):
        """Assign an id to the next shuffle operation."""
        self._rmp_shuffle_counter += 1
        return self._rmp_shuffle_counter

    def shuffle(self, df: dd.DataFrame, on: Sequence[str]):
        """Shuffle data using a RAPIDS-MP shuffle service."""
        client = get_client()
        shuffle_id = self.get_shuffle_id()
        meta = df._meta

        # Setup rapidsmp communicator
        initialize_ucxx_comms(client)

        # Extract mapping between ranks and worker addresses
        worker_ranks: dict[int, str] = {
            v: k for k, v in client.run(get_worker_rank).items()
        }
        n_workers = len(worker_ranks)
        restricted_keys: MutableMapping[Any, str] = {}

        # Add operation to submit each partition to the shuffler
        partition_count = df.optimize().npartitions
        df_id = df.map_partitions(
            rmp_shuffle_insert,
            on=on,
            partition_count=partition_count,
            shuffle_id=shuffle_id,
            meta=pd.DataFrame({"id": [shuffle_id]}),
            enforce_metadata=False,
        ).optimize()

        # Create task names
        token = tokenize(df_id, shuffle_id)
        global_barrier_name = f"rmp-global-barrier-{token}"
        worker_barrier_name = f"rmp-worker-barrier-{token}"
        extract_name = f"rmp-shuffle-extract-{token}"

        # Extract task graph and add global barrier task
        insert_keys = [(df_id._name, i) for i in range(df_id.npartitions)]
        dsk: MutableMapping[Any, Any] = {
            (global_barrier_name, 0): (
                global_rmp_barrier,
                shuffle_id,
                partition_count,
                insert_keys,
            )
        }
        dsk.update(df_id.dask)

        # Add worker barrier tasks
        worker_barriers: MutableMapping[Any, Any] = {}
        for rank, addr in worker_ranks.items():
            key = (worker_barrier_name, rank)
            worker_barriers[rank] = key
            dsk[key] = (
                worker_rmp_barrier,
                shuffle_id,
                partition_count,
                (global_barrier_name, 0),
            )
            restricted_keys[key] = addr

        # Add extraction tasks
        output_keys = []
        column_names = list(meta.columns)
        for part_id in range(partition_count):
            rank = part_id % n_workers
            output_keys.append((extract_name, part_id))
            dsk[output_keys[-1]] = (
                rmp_shuffle_extract,
                shuffle_id,
                part_id,
                column_names,
                worker_barriers[rank],
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

        # Construct/return a Dask-DataFrame collection
        divisions = (None,) * (partition_count + 1)
        name_prefix = "rmp-shuffle"
        return dd.from_graph(
            dsk,
            meta,
            divisions,
            output_keys,
            name_prefix,
        )


def dask_setup(scheduler):
    """Setup dask cluster."""
    plugin = RMPSchedulerPlugin(scheduler)
    scheduler.add_plugin(plugin)
