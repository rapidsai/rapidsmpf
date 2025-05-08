# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import dask
import dask.dataframe as dd
import pytest

import rapidsmpf.communicator
from rapidsmpf.examples.dask import DaskCudfIntegration
from rapidsmpf.integrations.dask.core import get_worker_context
from rapidsmpf.integrations.dask.shuffler import rapidsmpf_shuffle_graph
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
    if not rapidsmpf.communicator.MPI_SUPPORT:
        return False

    from mpi4py import MPI

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
        bootstrap_dask_cluster(client, spill_device=0.1)

        def get_rank(dask_worker: Worker) -> int:
            # TODO: maybe move the cast into rapidsmpf_comm?
            comm = get_worker_context(dask_worker).comm
            assert comm is not None
            return comm.rank

        result = client.run(get_rank)
        assert set(result.values()) == set(range(len(cluster.workers)))


@pytest.mark.parametrize("partition_count", [None, 3])
def test_dask_cudf_integration(loop: pytest.FixtureDef, partition_count: int) -> None:  # noqa: F811
    # Test basic Dask-cuDF integration
    pytest.importorskip("dask_cudf")

    import dask.dataframe as dd

    from rapidsmpf.examples.dask import dask_cudf_shuffle

    with LocalCUDACluster(loop=loop) as cluster:  # noqa: SIM117
        with Client(cluster) as client:
            bootstrap_dask_cluster(client, spill_device=0.1)
            df = (
                dask.datasets.timeseries(
                    freq="3600s",
                    partition_freq="2D",
                )
                .reset_index(drop=True)
                .to_backend("cudf")
            )
            partition_count_in = df.npartitions
            expect = df.compute().sort_values(["x", "y"])
            shuffled = dask_cudf_shuffle(
                df,
                ["name", "id"],
                partition_count=partition_count,
            )
            assert shuffled.npartitions == (partition_count or partition_count_in)
            got = shuffled.compute().sort_values(["x", "y"])

            dd.assert_eq(expect, got, check_index=False)


def test_bootstrap_dask_cluster_idempotent() -> None:
    with LocalCUDACluster() as cluster, Client(cluster) as client:
        bootstrap_dask_cluster(client, spill_device=0.1)
        before = client.run(
            lambda dask_worker: id(get_worker_context(dask_worker).comm)
        )

        bootstrap_dask_cluster(client, spill_device=0.1)
        after = client.run(lambda dask_worker: id(get_worker_context(dask_worker).comm))
        assert before == after


def test_boostrap_single_node_cluster_no_deadlock() -> None:
    with LocalCUDACluster(n_workers=1) as cluster, Client(cluster) as client:
        bootstrap_dask_cluster(client, spill_device=0.1)


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
                    column_names=column_names,
                    shuffle_on=shuffle_on,
                    partition_count_in=partition_count_in,
                    partition_count_out=partition_count_out,
                    integration=DaskCudfIntegration,
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
            bootstrap_dask_cluster(client, spill_device=0.1)
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
                match=f"Cannot shuffle more than {max_num_shuffles} times in a single Dask compute",
            ):
                do_shuffle(seed=3, num_shuffles=257)
