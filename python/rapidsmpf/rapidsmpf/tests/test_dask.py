# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import dask
import dask.dataframe as dd
import pytest
from distributed.diagnostics.plugin import WorkerPlugin

import rapidsmpf.integrations.single
from rapidsmpf.communicator import COMMUNICATORS
from rapidsmpf.config import Options
from rapidsmpf.examples.dask import DaskCudfIntegration, dask_cudf_shuffle
from rapidsmpf.integrations.dask.core import get_worker_context
from rapidsmpf.integrations.dask.shuffler import rapidsmpf_shuffle_graph
from rapidsmpf.shuffler import Shuffler, get_active_shuffle_ids

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


class DebugActiveShuffles(WorkerPlugin):
    def teardown(self, worker):  # type: ignore[no-untyped-def]
        print(f"Active shuffles {worker}:", get_active_shuffle_ids())


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
        client.register_plugin(DebugActiveShuffles())
        assert len(cluster.workers) == get_n_gpus()
        bootstrap_dask_cluster(client, options=Options({"dask_spill_device": "0.1"}))

        def get_rank(dask_worker: Worker) -> int:
            # TODO: maybe move the cast into rapidsmpf_comm?
            comm = get_worker_context(dask_worker).comm
            assert comm is not None
            return comm.rank

        result = client.run(get_rank)
        assert set(result.values()) == set(range(len(cluster.workers)))


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
            client.register_plugin(DebugActiveShuffles())
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
        client.register_plugin(DebugActiveShuffles())
        bootstrap_dask_cluster(client, options=options)
        before = client.run(
            lambda dask_worker: id(get_worker_context(dask_worker).comm)
        )

        bootstrap_dask_cluster(client, options=options)
        after = client.run(lambda dask_worker: id(get_worker_context(dask_worker).comm))
        assert before == after


def test_boostrap_single_node_cluster_no_deadlock() -> None:
    with LocalCUDACluster(n_workers=1) as cluster, Client(cluster) as client:
        client.register_plugin(DebugActiveShuffles())
        bootstrap_dask_cluster(client, options=Options({"dask_spill_device": "0.1"}))


def test_many_shuffles(loop: pytest.FixtureDef) -> None:  # noqa: F811
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
                rapidsmpf_shuffle_graph(
                    input_name=name_in,
                    output_name=name_out,
                    partition_count_in=partition_count_in,
                    partition_count_out=partition_count_out,
                    integration=DaskCudfIntegration,
                    options={"on": shuffle_on, "column_names": column_names},
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
            client.register_plugin(DebugActiveShuffles())
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
                    options={"on": shuffle_on, "column_names": column_names},
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
    ctx = rapidsmpf.integrations.single._get_worker_context()
    assert len(ctx.shufflers) == 0

    # But we cannot shuffle more than `max_num_shuffles` times in a single compute.
    with pytest.raises(
        ValueError,
        match=f"Cannot shuffle more than {max_num_shuffles} times in a single query",
    ):
        do_shuffle(seed=3, num_shuffles=max_num_shuffles + 1)
