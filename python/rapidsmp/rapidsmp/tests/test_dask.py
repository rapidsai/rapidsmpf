# Copyright (c) 2025, NVIDIA CORPORATION.
from __future__ import annotations

from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from dask_cuda.utils import get_n_gpus
from distributed.utils_test import gen_test

from rapidsmp.integrations.dask import rapidsmp_ucxx_comm_setup


@gen_test(timeout=20)
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
