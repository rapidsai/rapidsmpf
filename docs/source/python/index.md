# Python

RapidsMPF provides a Python API for building high-performance multi-GPU data pipelines.
The Python layer wraps the C++ core and integrates with popular distributed computing
frameworks.

## Quickstart

- {doc}`quickstart` — Dask-cuDF shuffle example and Streaming Engine example

## API Reference

- {doc}`api` — Full Python API reference (integrations, shuffler, communicator, memory, config)

## Integrations

The Python API includes ready-to-use integrations with:

- **Dask** (`rapidsmpf.integrations.dask`) — shuffle Dask DataFrames across a
  `LocalCUDACluster` or multi-node Dask deployment.
- **Ray** (`rapidsmpf.integrations.ray`) — use RapidsMPF within Ray tasks and actors.
- **Single-process** (`rapidsmpf.integrations.single`) — run multi-GPU workloads in a
  single Python process without a cluster manager.
- **cuDF** (`rapidsmpf.integrations.cudf`) — partition and pack/unpack cuDF tables for
  use with the Shuffler.

```{toctree}
---
maxdepth: 1
hidden:
---
quickstart.md
api.md
```
