# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import pynvml
import toolz

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"

import pytest

ray = pytest.importorskip("ray")

from rapidsmp.examples.ray.ray_shuffle_example import (  # noqa: E402
    ShufflingActor,
)
from rapidsmp.integrations.ray import (  # noqa: E402
    RapidsMPActor,
    setup_ray_ucxx_cluster,
)


def get_nranks_if_spawned_by_mpi() -> int:
    """Check if running on an MPI env without importing MPI libs"""
    mpi_env_vars = [
        "OMPI_COMM_WORLD_SIZE",  # OpenMPI
        "PMI_SIZE",  # MPICH/SLURM
        "MPI_WORLD_SIZE",  # Intel MPI
        "MV2_COMM_WORLD_SIZE",  # MVAPICH
    ]

    for var in mpi_env_vars:
        if var in os.environ:
            return int(os.environ[var])

    return -1


pytestmark = pytest.mark.skipif(
    get_nranks_if_spawned_by_mpi() > 1,
    reason="Ray tests should not run with more than one MPI process",
)

# initialize ray with 4 cpu processes
ray.init(num_cpus=4)


@ray.remote(num_cpus=1)
class DummyActor(RapidsMPActor):
    def use_comm(self) -> None:
        # test if the DummyActor can use the Communicator object
        assert self.comm is not None  # for mypy
        assert all(
            sub in self.comm.get_str()
            for sub in ["UCXX", f"rank={self._rank}", f"nranks={self._nranks}"]
        )


@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_ray_ucxx_cluster(num_workers: int) -> None:
    # setup the UCXX cluster using DummyActors
    gpu_actors = setup_ray_ucxx_cluster(DummyActor, num_workers)

    try:
        # get the rank of each actor remotely
        ranks = ray.get([actor.rank.remote() for actor in gpu_actors])

        # ranks should be [0...num_workers-1]
        assert set(ranks) == set(range(num_workers))

        # test if the DummyActor can use the Communicator object
        ray.get([actor.use_comm.remote() for actor in gpu_actors])
    finally:
        for actor in gpu_actors:
            ray.kill(actor)


@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_ray_ucxx_cluster_not_initialized(num_workers: int) -> None:
    # setup the UCXX cluster using DummyActors
    # there's some fancy monkeypatching in ray making `DummyActor.remote` available
    gpu_actors = [DummyActor.remote(num_workers) for _ in range(num_workers)]  # type: ignore[attr-defined]

    try:
        # all actors should not be initialized
        init_flags = ray.get([actor.is_initialized.remote() for actor in gpu_actors])
        assert not all(init_flags)

        with pytest.raises(ray.exceptions.RayTaskError):
            # this will fail because ray is not initialized
            ray.get([actor.to_string.remote() for actor in gpu_actors])
    finally:
        for actor in gpu_actors:
            ray.kill(actor)


def test_disallowed_classes() -> None:
    # class that doesnt extend RapidsMPActor or ray actor
    class NonActor: ...

    with pytest.raises(TypeError):
        setup_ray_ucxx_cluster(NonActor, 1)

    # class that only extends RapidsMPActor
    class NonRayActor(RapidsMPActor): ...

    with pytest.raises(TypeError):
        setup_ray_ucxx_cluster(NonRayActor, 1)

    # class that only extends ray actor
    @ray.remote(num_cpus=1)
    class NonRapidsMPActor: ...

    with pytest.raises(TypeError):
        setup_ray_ucxx_cluster(NonRapidsMPActor, 1)


@toolz.memoize
def get_gpu_count() -> int:
    pynvml.nvmlInit()
    device_count = int(pynvml.nvmlDeviceGetCount())  # casting Any to int
    pynvml.nvmlShutdown()
    return device_count


@pytest.mark.parametrize("num_workers", [1, 4])
@pytest.mark.parametrize("batch_size", [-1, 10])
@pytest.mark.parametrize("total_num_partitions", [1, 10])
def test_ray_shuffle_actor(
    num_workers: int, batch_size: int, total_num_partitions: int
) -> None:
    gpu_count = get_gpu_count()

    # Test shuffling actor that uses 1/num_workers fractional GPUs if
    # gpu_count < num_workers or 1 GPU otherwise
    @ray.remote(num_gpus=(gpu_count / num_workers) if gpu_count < num_workers else 1)
    class TestShufflingActor(ShufflingActor): ...

    # setup the UCXX cluster using TestShufflingActor
    gpu_actors = setup_ray_ucxx_cluster(
        TestShufflingActor,
        num_workers,
        batch_size=batch_size,
        total_nparts=total_num_partitions,
    )

    try:
        # call run on all actors remotely
        ray.get([actor.run.remote() for actor in gpu_actors])
    finally:
        for actor in gpu_actors:
            ray.kill(actor)
