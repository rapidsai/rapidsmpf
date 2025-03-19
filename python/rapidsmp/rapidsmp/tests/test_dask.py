# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import dask
import pytest

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
from mpi4py import MPI  # noqa: E402

from rapidsmp.integrations.dask import (  # noqa: E402
    bootstrap_dask_cluster,
    bootstrap_dask_cluster_async,
)

if TYPE_CHECKING:
    from distributed.worker import Worker


def get_mpi_nsize() -> int:
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    # without the cast, mypy sees this as Any
    return cast(int, size)


pytestmark = pytest.mark.skipif(
    get_mpi_nsize() > 1,
    reason="Dask tests should not run with more than one MPI process",
)


@gen_test(timeout=30)
async def test_dask_ucxx_cluster_sync() -> None:
    with (
        LocalCUDACluster(scheduler_port=0, device_memory_limit=1) as cluster,
        Client(cluster) as client,
    ):
        assert len(cluster.workers) == get_n_gpus()
        bootstrap_dask_cluster(client, pool_size=0.25, spill_device=0.1)

        def get_rank(dask_worker: Worker) -> int:
            # TODO: maybe move the cast into rapidsmp_comm?
            return cast(int, dask_worker._rapidsmp_comm.rank)

        result = client.run(get_rank)
        assert set(result.values()) == set(range(len(cluster.workers)))


@gen_test(timeout=30)
async def test_dask_ucxx_cluster_async() -> None:
    async with (
        LocalCUDACluster(
            scheduler_port=0, asynchronous=True, device_memory_limit=1
        ) as cluster,
        Client(cluster, asynchronous=True) as client,
    ):
        assert len(cluster.workers) == get_n_gpus()
        await bootstrap_dask_cluster_async(client, pool_size=0.25, spill_device=0.1)

        def get_rank(dask_worker: Worker) -> int:
            # TODO: maybe move the cast into rapidsmp_comm?
            return cast(int, dask_worker._rapidsmp_comm.rank)

        result = await client.run(get_rank)
        assert set(result.values()) == set(range(len(cluster.workers)))


@pytest.mark.parametrize("partition_count", [None, 3])
def test_dask_cudf_integration(loop: pytest.FixtureDef, partition_count: int) -> None:  # noqa: F811
    # Test basic Dask-cuDF integration
    pytest.importorskip("dask_cudf")

    import dask.dataframe as dd

    from rapidsmp.examples.dask import dask_cudf_shuffle

    with LocalCUDACluster(loop=loop) as cluster:  # noqa: SIM117
        with Client(cluster) as client:
            bootstrap_dask_cluster(client, pool_size=0.25, spill_device=0.1)
            print("Cluster ready!")
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
        bootstrap_dask_cluster(client, pool_size=0.25, spill_device=0.1)
        bootstrap_dask_cluster(client, pool_size=0.25, spill_device=0.1)


@gen_test(timeout=30)
async def test_bootstrap_dask_cluster_async_idempotent() -> None:
    async with (
        LocalCUDACluster(asynchronous=True) as cluster,
        Client(cluster, asynchronous=True) as client,
    ):
        await bootstrap_dask_cluster_async(client, pool_size=0.25, spill_device=0.1)
        await bootstrap_dask_cluster_async(client, pool_size=0.25, spill_device=0.1)


def test_boostrap_single_node_cluster_no_deadlock() -> None:
    with LocalCUDACluster(n_workers=1) as cluster, Client(cluster) as client:
        bootstrap_dask_cluster(client, pool_size=0.25, spill_device=0.1)
