# Copyright (c) 2025, NVIDIA CORPORATION.
from __future__ import annotations

import dask
import pytest

dask_cuda = pytest.importorskip("dask_cuda")

from dask.distributed import Client  # noqa: E402
from dask_cuda import LocalCUDACluster  # noqa: E402
from dask_cuda.utils import get_n_gpus  # noqa: E402
from distributed.utils_test import cleanup, gen_test, loop, loop_in_thread # noqa: E402
from mpi4py import MPI  # noqa: E402

from rapidsmp.integrations.dask import rapidsmp_ucxx_comm_setup  # noqa: E402


def get_mpi_nsize():
    comm = MPI.COMM_WORLD
    return comm.Get_size()


pytestmark = pytest.mark.skipif(
    get_mpi_nsize() > 1,
    reason="Dask tests should not run with more than one MPI process",
)


@gen_test(timeout=30)
async def test_dask_ucxx_cluster():
    async with (
        LocalCUDACluster(
            scheduler_port=0, asynchronous=True, device_memory_limit=1
        ) as cluster,
        Client(cluster, asynchronous=True) as client,
    ):
        assert len(cluster.workers) == get_n_gpus()

        await rapidsmp_ucxx_comm_setup(client)

        def get_rank(dask_worker):
            return dask_worker._rapidsmp_comm.rank

        result = await client.run(get_rank)
        assert set(result.values()) == set(range(len(cluster.workers)))


@pytest.mark.parametrize("partition_count", [None, 3])
def test_dask_cudf_integration(loop, partition_count):  # noqa: F811
    # Test basic Dask-cuDF integration

    pytest.importorskip("dask_cudf")

    import dask.dataframe as dd

    from rapidsmp.examples.dask import dask_cudf_shuffle
    from rapidsmp.integrations.dask import LocalRMPCluster

    with LocalRMPCluster(loop=loop) as cluster:  # noqa: SIM117
        with Client(cluster):
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
