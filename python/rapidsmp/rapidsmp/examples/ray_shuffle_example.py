# Copyright (c) 2025, NVIDIA CORPORATION.
"""Example running a RapidsMP Shuffle operation using Ray and UCXX communication."""

from __future__ import annotations

import argparse

import ray

from rapidsmp.integrations.ray import (
    ShufflingActor,
    setup_ray_ucxx_cluster,
)


@ray.remote(num_gpus=1)
class ShuffleExample(ShufflingActor):
    """Shuffle example class with 1 GPU resource."""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RapidsMP Ray Shuffling Actor example ",
    )
    parser.add_argument("--nranks", type=int, default=1)
    parser.add_argument("--num_rows", type=int, default=100)
    parser.add_argument("--batch_sz", type=int, default=-1)
    parser.add_argument("--total_nparts", type=int, default=-1)
    args = parser.parse_args()

    ray.init()  # init ray with all resources

    # Create shufflling actors
    gpu_actors = setup_ray_ucxx_cluster(
        ShuffleExample, args.nranks, args.num_rows, args.batch_sz, args.total_nparts
    )

    try:
        # run the ShufflingActor.run method remotely
        ray.get([actor.run.remote() for actor in gpu_actors])  # type: ignore

    finally:
        for actor in gpu_actors:
            ray.kill(actor)
