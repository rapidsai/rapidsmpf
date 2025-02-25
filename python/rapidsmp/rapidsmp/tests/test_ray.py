# Copyright (c) 2025, NVIDIA CORPORATION.
from __future__ import annotations

import os

import pytest
import ray

from rapidsmp.integrations.ray import RapidsMPActor, setup_ray_ucx_cluster

os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"

# initialize ray with 4 cpu processes
ray.init(num_cpus=4)


@ray.remote(num_cpus=1)
class DummyActor(RapidsMPActor):
    def __init__(self, rank, num_workers):
        super().__init__(rank, num_workers)


@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_ray_ucx_cluster(num_workers):
    # setup the ucx cluster using DummyActors
    gpu_actors = setup_ray_ucx_cluster(DummyActor, num_workers)

    # call DummyActor.status_check() method remotely
    ray.get([actor.status_check.remote() for actor in gpu_actors])

    for actor in gpu_actors:
        ray.kill(actor)


@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_ray_ucx_cluster_not_initialized(num_workers):
    # setup the ucx cluster using DummyActors
    gpu_actors = [DummyActor.remote(i, num_workers) for i in range(num_workers)]

    # all actors should not be initialized
    init_flags = ray.get([actor.is_initialized.remote() for actor in gpu_actors])
    assert not all(init_flags)

    with pytest.raises(ray.exceptions.RayTaskError):
        # this will fail because ray is not initialized
        ray.get([actor.status_check.remote() for actor in gpu_actors])

    for actor in gpu_actors:
        ray.kill(actor)
