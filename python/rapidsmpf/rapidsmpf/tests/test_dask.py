# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import multiprocessing
from typing import TYPE_CHECKING, Literal

import dask
import dask.dataframe as dd
import pytest

import rmm.mr

import rapidsmpf.integrations.single
from rapidsmpf.communicator import COMMUNICATORS
from rapidsmpf.config import Options
from rapidsmpf.examples.dask import (
    DaskCudfIntegration,
    dask_cudf_join,
    dask_cudf_shuffle,
)
from rapidsmpf.integrations.core import setup_rmm_pool
from rapidsmpf.integrations.dask.core import get_worker_context
from rapidsmpf.integrations.dask.shuffler import (
    clear_shuffle_statistics,
    gather_shuffle_statistics,
    rapidsmpf_shuffle_graph,
)
from rapidsmpf.shuffler import Shuffler

dask_cuda = pytest.importorskip("dask_cuda")

from dask.distributed import Client  # noqa: E402
from dask_cuda import LocalCUDACluster  # noqa: E402
from dask_cuda.utils import get_n_gpus  # noqa: E402
from distributed.utils_test import (  # noqa: E402, F401
    cleanup,
    gen_test,
    loop,
    loop_in_thread,
)

from rapidsmpf.integrations.dask.core import (  # noqa: E402
    bootstrap_dask_cluster,
)

if TYPE_CHECKING:
    from distributed.worker import Worker


def is_running_on_multiple_mpi_nodes() -> bool:
    if "mpi" not in COMMUNICATORS:
        return False

    MPI = pytest.importorskip("mpi4py.MPI")

    return bool(MPI.COMM_WORLD.Get_size() > 1)


pytestmark = pytest.mark.skipif(
    is_running_on_multiple_mpi_nodes(),
    reason="Dask tests should not run with more than one MPI process",
)


@gen_test(timeout=30)
async def test_dask_ucxx_cluster_sync() -> None:
    with (
        LocalCUDACluster(scheduler_port=0, device_memory_limit=1) as cluster,
        Client(cluster) as client,
    ):
        assert len(cluster.workers) == get_n_gpus()
        bootstrap_dask_cluster(client, options=Options({"dask_spill_device": "0.1"}))

        def get_rank(dask_worker: Worker) -> int:
            # TODO: maybe move the cast into rapidsmpf_comm?
            comm = get_worker_context(dask_worker).comm
            assert comm is not None
            return comm.rank

        result = client.run(get_rank)
        assert set(result.values()) == set(range(len(cluster.workers)))


def assert_workers_contain_mr_type(
    client: Client, resource_type: type[rmm.mr.MemoryResource]
) -> None:
    """Assert that all workers contain the given resource type."""

    def _has_rmm_resource_type() -> bool:
        # The given resource type must be in the resource chain.
        mr = rmm.mr.get_current_device_resource()
        while mr is not None:
            if isinstance(mr, resource_type):
                return True
            mr = getattr(mr, "upstream_mr", None)
        return False

    result = client.run(_has_rmm_resource_type)
    for worker_addr, has_type in result.items():
        assert has_type, (
            f"Worker {worker_addr} does not have {resource_type.__name__} in chain."
        )


@gen_test(timeout=30)
async def test_dask_rmm_setup_async() -> None:
    """Test that rapidsmpf can set up RMM async pool during bootstrap."""
    with (
        LocalCUDACluster(
            scheduler_port=0,
            device_memory_limit=1,
            # Don't let dask-cuda set up RMM - rapidsmpf will do it
            rmm_pool_size=None,
            rmm_async=False,
        ) as cluster,
        Client(cluster) as client,
    ):
        bootstrap_dask_cluster(
            client,
            options=Options(
                {
                    "dask_spill_device": "0.1",
                    "dask_rmm_async": "true",
                    "dask_rmm_pool_size": "0.1",  # 10% of device memory
                }
            ),
        )
        assert_workers_contain_mr_type(client, rmm.mr.CudaAsyncMemoryResource)


@gen_test(timeout=30)
async def test_dask_rmm_setup_pool() -> None:
    """Test that rapidsmpf can set up RMM pool (non-async) during bootstrap."""
    with (
        LocalCUDACluster(
            scheduler_port=0,
            device_memory_limit=1,
            # Don't let dask-cuda set up RMM - rapidsmpf will do it
            rmm_pool_size=None,
            rmm_async=False,
        ) as cluster,
        Client(cluster) as client,
    ):
        bootstrap_dask_cluster(
            client,
            options=Options(
                {
                    "dask_spill_device": "0.1",
                    "dask_rmm_async": "false",
                    "dask_rmm_pool_size": "0.1",  # 10% of device memory
                }
            ),
        )

        assert_workers_contain_mr_type(client, rmm.mr.PoolMemoryResource)


@gen_test(timeout=30)
async def test_dask_rmm_setup_track_allocations() -> None:
    """Test that rapidsmpf can enable RMM tracking during bootstrap."""
    with (
        LocalCUDACluster(
            scheduler_port=0,
            device_memory_limit=1,
            rmm_pool_size=None,
            rmm_async=False,
        ) as cluster,
        Client(cluster) as client,
    ):
        bootstrap_dask_cluster(
            client,
            options=Options(
                {
                    "dask_spill_device": "0.1",
                    "dask_rmm_track_allocations": "true",
                }
            ),
        )

        assert_workers_contain_mr_type(client, rmm.mr.TrackingResourceAdaptor)


def test_rmm_setup_validation_errors() -> None:
    """Test that invalid RMM option combinations raise errors."""

    # rmm_managed_memory is incompatible with rmm_async
    with pytest.raises(ValueError, match="incompatible"):
        setup_rmm_pool(
            "test_",
            Options(
                {
                    "test_rmm_async": "true",
                    "test_rmm_managed_memory": "true",
                }
            ),
        )

    # rmm_release_threshold requires rmm_async
    with pytest.raises(ValueError, match="requires"):
        setup_rmm_pool(
            "test_",
            Options(
                {
                    "test_rmm_async": "false",
                    "test_rmm_pool_size": "0.1",  # 10% of device memory
                    "test_rmm_release_threshold": "0.05",  # 5% of device memory
                }
            ),
        )


@pytest.mark.parametrize("partition_count", [None, 3])
@pytest.mark.parametrize("sort", [True, False])
def test_dask_cudf_integration(
    loop: pytest.FixtureDef,  # noqa: F811
    partition_count: int,
    sort: bool,  # noqa: FBT001
) -> None:
    # Test basic Dask-cuDF integration
    pytest.importorskip("dask_cudf")

    with LocalCUDACluster(loop=loop) as cluster:  # noqa: SIM117
        with Client(cluster) as client:
            bootstrap_dask_cluster(
                client, options=Options({"dask_spill_device": "0.1"})
            )
            df = (
                dask.datasets.timeseries(
                    freq="3600s",
                    partition_freq="2D",
                )
                .reset_index(drop=True)
                .to_backend("cudf")
            )
            partition_count_in = df.npartitions
            expect = df.compute().sort_values(["id", "name", "x", "y"])
            shuffled = dask_cudf_shuffle(
                df,
                ["id", "name"],
                sort=sort,
                partition_count=partition_count,
            )
            assert shuffled.npartitions == (partition_count or partition_count_in)
            got = shuffled.compute()
            if sort:
                assert got["id"].is_monotonic_increasing
            got = got.sort_values(["id", "name", "x", "y"])

            dd.assert_eq(expect, got, check_index=False)


@pytest.mark.parametrize("partition_count", [None, 3])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("cluster_kind", ["auto", "single"])
def test_dask_cudf_integration_single(
    partition_count: int,
    sort: bool,  # noqa: FBT001
    cluster_kind: Literal["distributed", "single", "auto"],
) -> None:
    # Test single-worker cuDF integration with Dask-cuDF
    pytest.importorskip("dask_cudf")

    df = (
        dask.datasets.timeseries(
            freq="3600s",
            partition_freq="2D",
        )
        .reset_index(drop=True)
        .to_backend("cudf")
    )
    partition_count_in = df.npartitions
    expect = df.compute().sort_values(["id", "name", "x", "y"])
    shuffled = dask_cudf_shuffle(
        df,
        ["id", "name"],
        sort=sort,
        partition_count=partition_count,
        cluster_kind=cluster_kind,
        config_options=Options({"single_spill_device": "0.1"}),
    )
    assert shuffled.npartitions == (partition_count or partition_count_in)
    got = shuffled.compute()
    if sort:
        assert got["id"].is_monotonic_increasing
    got = got.sort_values(["id", "name", "x", "y"])

    dd.assert_eq(expect, got, check_index=False)


def test_dask_cudf_integration_single_raises() -> None:
    pytest.importorskip("dask_cudf")

    from rapidsmpf.examples.dask import dask_cudf_shuffle

    df = dask.datasets.timeseries().reset_index(drop=True).to_backend("cudf")
    with pytest.raises(ValueError, match="No global client"):
        dask_cudf_shuffle(df, ["id", "name"], cluster_kind="distributed")
    with pytest.raises(ValueError, match="Expected one of"):
        dask_cudf_shuffle(df, ["id", "name"], cluster_kind="foo")  # type: ignore


def test_bootstrap_dask_cluster_idempotent() -> None:
    options = Options({"dask_spill_device": "0.1"})
    with LocalCUDACluster() as cluster, Client(cluster) as client:
        bootstrap_dask_cluster(client, options=options)
        before = client.run(
            lambda dask_worker: id(get_worker_context(dask_worker).comm)
        )

        bootstrap_dask_cluster(client, options=options)
        after = client.run(lambda dask_worker: id(get_worker_context(dask_worker).comm))
        assert before == after


def test_boostrap_single_node_cluster_no_deadlock() -> None:
    with LocalCUDACluster(n_workers=1) as cluster, Client(cluster) as client:
        bootstrap_dask_cluster(client, options=Options({"dask_spill_device": "0.1"}))


def test_many_shuffles(loop: pytest.FixtureDef) -> None:  # noqa: F811
    pytest.importorskip("dask_cudf")

    def clear_shuffles(dask_worker: Worker) -> None:
        # Avoid leaking Shuffler objects between tests, by clearing
        # finished shuffles and shutting down (and clearing) staged,
        # but not finished, suffles. This shouldn't hang because in the
        # "too many shuffles" case, we just stage shuffles without actually
        # inserting (or extracting) any data, and so shutdown shouldn't block.
        ctx = get_worker_context(dask_worker)
        for shuffle_id, shuffler in list(ctx.shufflers.items()):
            if ctx.shufflers[shuffle_id].finished():
                del ctx.shufflers[shuffle_id]
            else:
                shuffler.shutdown()
                del ctx.shufflers[shuffle_id]

    def do_shuffle(seed: int, num_shuffles: int) -> None:
        """Shuffle a dataframe `num_shuffles` consecutive times and check the result"""
        expect = (
            dask.datasets.timeseries(
                freq="3600s",
                partition_freq="2D",
                seed=seed,
            )
            .reset_index(drop=True)
            .to_backend("cudf")
        )
        df0 = expect.optimize()
        partition_count_in = df0.npartitions
        partition_count_out = partition_count_in
        column_names = list(df0.columns)
        shuffle_on = ["name", "id"]

        graph = df0.dask.copy()
        name_in = df0._name
        for i in range(num_shuffles):
            name_out = f"test_many_shuffles-output-{i}"
            graph.update(
                rapidsmpf_shuffle_graph(
                    input_name=name_in,
                    output_name=name_out,
                    partition_count_in=partition_count_in,
                    partition_count_out=partition_count_out,
                    integration=DaskCudfIntegration,
                    options={
                        "on": shuffle_on,
                        "column_names": column_names,
                        "cluster_kind": "distributed",
                    },
                )
            )
            name_in = name_out
        got = dd.from_graph(
            graph,
            df0._meta,
            (None,) * (partition_count_out + 1),
            [(name_out, pid) for pid in range(partition_count_out)],
            "rapidsmpf",
        )
        dd.assert_eq(
            expect.compute().sort_values(["x", "y"]),
            got.compute().sort_values(["x", "y"]),
            check_index=False,
        )

    with LocalCUDACluster(n_workers=1, loop=loop) as cluster:  # noqa: SIM117
        with Client(cluster) as client:
            bootstrap_dask_cluster(
                client, options=Options({"dask_spill_device": "0.1"})
            )
            max_num_shuffles = Shuffler.max_concurrent_shuffles

            # We can shuffle `max_num_shuffles` consecutive times.
            do_shuffle(seed=1, num_shuffles=max_num_shuffles)
            # And more times after a compute.
            do_shuffle(seed=2, num_shuffles=10)

            # Check that all shufflers has been cleaned up.
            def check_worker(dask_worker: Worker) -> None:
                ctx = get_worker_context(dask_worker)
                assert len(ctx.shufflers) == 0

            client.run(check_worker)

            # But we cannot shuffle more than `max_num_shuffles` times in a single compute.
            with pytest.raises(
                ValueError,
                match=f"Cannot shuffle more than {max_num_shuffles} times in a single query",
            ):
                do_shuffle(seed=3, num_shuffles=max_num_shuffles + 1)

            client.run(clear_shuffles)


def test_many_shuffles_single() -> None:
    pytest.importorskip("dask_cudf")

    def do_shuffle(seed: int, num_shuffles: int) -> None:
        """Shuffle a dataframe `num_shuffles` consecutive times and check the result"""
        expect = (
            dask.datasets.timeseries(
                freq="3600s",
                partition_freq="2D",
                seed=seed,
            )
            .reset_index(drop=True)
            .to_backend("cudf")
        )
        df0 = expect.optimize()
        partition_count_in = df0.npartitions
        partition_count_out = partition_count_in
        column_names = list(df0.columns)
        shuffle_on = ["name", "id"]

        graph = df0.dask.copy()
        name_in = df0._name
        for i in range(num_shuffles):
            name_out = f"test_many_shuffles-output-{i}"
            graph.update(
                rapidsmpf.integrations.single.rapidsmpf_shuffle_graph(
                    input_name=name_in,
                    output_name=name_out,
                    partition_count_in=partition_count_in,
                    partition_count_out=partition_count_out,
                    integration=DaskCudfIntegration,
                    options={
                        "on": shuffle_on,
                        "column_names": column_names,
                        "cluster_kind": "single",
                    },
                )
            )
            name_in = name_out
        got = dd.from_graph(
            graph,
            df0._meta,
            (None,) * (partition_count_out + 1),
            [(name_out, pid) for pid in range(partition_count_out)],
            "rapidsmpf",
        )
        dd.assert_eq(
            expect.compute().sort_values(["x", "y"]),
            got.compute().sort_values(["x", "y"]),
            check_index=False,
        )

    rapidsmpf.integrations.single.setup_worker(
        options=Options({"single_spill_device": "0.1"})
    )
    max_num_shuffles = Shuffler.max_concurrent_shuffles

    # We can shuffle `max_num_shuffles` consecutive times.
    do_shuffle(seed=1, num_shuffles=max_num_shuffles)
    # And more times after a compute.
    do_shuffle(seed=2, num_shuffles=10)

    # Check that all shufflers has been cleaned up.
    ctx = rapidsmpf.integrations.single.get_worker_context()
    assert len(ctx.shufflers) == 0

    # But we cannot shuffle more than `max_num_shuffles` times in a single compute.
    with pytest.raises(
        ValueError,
        match=f"Cannot shuffle more than {max_num_shuffles} times in a single query",
    ):
        do_shuffle(seed=3, num_shuffles=max_num_shuffles + 1)

    # Cleanup Shufflers to avoid leaking between tests.
    # This shouldn't hang because we just stage shuffles without,
    # without inserting or extracting any data, and so shutdown shouldn't block.
    context = rapidsmpf.integrations.single.get_worker_context()
    for shuffle_id, shuffler in list(context.shufflers.items()):
        if context.shufflers[shuffle_id].finished():
            del context.shufflers[shuffle_id]
        else:
            shuffler.shutdown()
            del context.shufflers[shuffle_id]


def test_gather_shuffle_statistics() -> None:
    with LocalCUDACluster(n_workers=1) as cluster, Client(cluster) as client:
        config_options = Options({"dask_statistics": "true"})

        df = dask.datasets.timeseries().reset_index(drop=True).to_backend("cudf")
        shuffled = dask_cudf_shuffle(df, on=["name"], config_options=config_options)
        shuffled.compute()

        stats = gather_shuffle_statistics(client)
        expected_stats = {
            "event-loop-check-future-finish",
            "event-loop-init-gpu-data-send",
            "event-loop-metadata-recv",
            "event-loop-metadata-send",
            "event-loop-post-incoming-chunk-recv",
            "event-loop-total",
            "shuffle-payload-recv",
            "shuffle-payload-send",
            "spill-bytes-host-to-device",
            "spill-time-host-to-device",
        }

        assert set(stats) == expected_stats
        for stat in expected_stats:
            assert stats[stat]["count"] > 0
            assert "value" in stats[stat]

        assert stats["shuffle-payload-send"]["value"] > 0
        assert (
            stats["shuffle-payload-send"]["value"]
            == stats["shuffle-payload-recv"]["value"]
        )


def test_clear_shuffle_statistics() -> None:
    with LocalCUDACluster(n_workers=1) as cluster, Client(cluster) as client:
        config_options = Options(
            {"dask_statistics": "true", "dask_print_statistics": "false"}
        )

        df = dask.datasets.timeseries().reset_index(drop=True).to_backend("cudf")
        shuffled = dask_cudf_shuffle(df, on=["name"], config_options=config_options)
        shuffled.compute()

        stats = gather_shuffle_statistics(client)
        assert len(stats) > 0

        clear_shuffle_statistics(client)
        stats2 = gather_shuffle_statistics(client)

        assert len(stats) > 0
        assert len(stats2) == 0


@pytest.mark.parametrize("how", ["inner", "left", "right"])
@pytest.mark.parametrize("left_pre_shuffled", [True, False])
@pytest.mark.parametrize("right_pre_shuffled", [True, False])
def test_dask_cudf_join(
    loop: pytest.FixtureDef,  # noqa: F811
    how: Literal["inner", "left", "right"],
    left_pre_shuffled: bool,  # noqa: FBT001
    right_pre_shuffled: bool,  # noqa: FBT001
) -> None:
    # Test basic Dask-cuDF unified join integration
    pytest.importorskip("dask_cudf")

    with LocalCUDACluster(loop=loop) as cluster:  # noqa: SIM117
        with Client(cluster) as client:
            bootstrap_dask_cluster(
                client, options=Options({"dask_spill_device": "0.1"})
            )
            left0 = (
                dask.datasets.timeseries(
                    freq="3600s",
                    partition_freq="2D",
                )
                .reset_index(drop=True)
                .to_backend("cudf")
            )
            right0 = (
                dask.datasets.timeseries(
                    freq="360s",
                    partition_freq="15D",
                )
                .reset_index(drop=True)
                .to_backend("cudf")
                .rename(
                    columns={
                        "id": "id2",
                        "name": "name2",
                        "x": "x2",
                        "y": "y2",
                    }
                )
            )
            left_on = ["id", "name"]
            right_on = ["id2", "name2"]

            # Maybe pre-shuffle the inputs
            left = (
                dask_cudf_shuffle(
                    left0,
                    left_on,
                    partition_count=max(left0.npartitions, right0.npartitions),
                    cluster_kind="distributed",
                )
                if left_pre_shuffled
                else left0
            )
            right = (
                dask_cudf_shuffle(
                    right0,
                    right_on,
                    partition_count=max(left0.npartitions, right0.npartitions),
                    cluster_kind="distributed",
                )
                if right_pre_shuffled
                else right0
            )

            # Join the inputs
            joined = dask_cudf_join(
                left,
                right,
                left_on=left_on,
                right_on=right_on,
                how=how,
                left_pre_shuffled=left_pre_shuffled,
                right_pre_shuffled=right_pre_shuffled,
            ).compute()

            # Check the result.
            # NOTE: We cannot call compute on a collection containing
            # a RMPF shuffle multiple times. Therefore, we use left0
            # and right0 to generate the expected result (left and
            # right may be "pre-shuffled").
            expected = left0.merge(
                right0, left_on=left_on, right_on=right_on, how=how
            ).compute()
            dd.assert_eq(joined, expected, check_index=False)


@gen_test(timeout=30)
@pytest.mark.filterwarnings("ignore")
async def test_bootstrap_multiple_clients(
    loop: pytest.FixtureDef,  # noqa: F811
) -> None:
    # https://github.com/rapidsai/rapidsmpf/issues/458

    def connect_from_subprocess(
        scheduler_address: str, q: multiprocessing.Queue
    ) -> None:
        client = Client(scheduler_address)
        bootstrap_dask_cluster(client)
        q.put(obj=True)

    with LocalCUDACluster(loop=loop) as cluster:
        with Client(cluster) as client_1:
            bootstrap_dask_cluster(client_1)

        with Client(cluster) as client_2:
            bootstrap_dask_cluster(client_2)

        q: multiprocessing.Queue[bool] = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=connect_from_subprocess, args=(cluster.scheduler_address, q)
        )
        p.start()
        result = q.get(timeout=10)
        p.join()

    assert result is True
