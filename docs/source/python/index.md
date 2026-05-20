# Python

RapidsMPF provides a Python API for building high-performance multi-GPU data pipelines.
The Python layer wraps the C++ core and integrates with popular distributed computing
frameworks.

## Quickstart

- {doc}`quickstart` — Streaming Engine example

## API Reference

- {doc}`api` — Full Python API reference (integrations, shuffler, communicator, memory, config)

## Integrations

The Python API includes ready-to-use integrations with:

- **Ray** (`rapidsmpf.integrations.ray`) — use RapidsMPF within Ray tasks and actors.
- **cuDF** (`rapidsmpf.integrations.cudf`) — partition and pack/unpack cuDF tables for
  use with the Shuffler.

```{toctree}
---
maxdepth: 2
hidden:
---
quickstart.md
api.md
```
