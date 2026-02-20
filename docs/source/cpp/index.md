# C++

RapidsMPF exposes a full C++ API for building high-performance distributed GPU
workloads without a Python runtime. The C++ layer is the foundation on which the
Python bindings are built.

The C++ API reference is available at
[docs.rapids.ai/api/librapidsmpf/stable](https://docs.rapids.ai/api/librapidsmpf/stable/)
([nightly](https://docs.rapids.ai/api/librapidsmpf/nightly/)).

## Coverage

The C++ API provides access to all core RapidsMPF subsystems:

- **Communicator** — MPI and UCXX backends for inter-process communication.
- **Shuffler** — Out-of-core, distributed table shuffle service.
- **Streaming Engine** — Asynchronous multi-GPU pipeline with Channels, Actors, and Messages.
- **Memory** — BufferResource, spilling, pinned memory, and packed data utilities.
- **Config** — Configuration options and environment-variable parsing.

## Table Shuffle Service

See {doc}`../background/shuffle-architecture` for an in-depth explanation of the
shuffle design.

The following is a complete MPI program that uses the RapidsMPF shuffler:

```{literalinclude} ../../../cpp/examples/example_shuffle.cpp
:language: cpp
:lines: 7-
```

## rrun — Distributed Launcher

RapidsMPF includes `rrun`, a lightweight launcher that eliminates the MPI dependency
for multi-GPU workloads. See {doc}`../background/streaming-engine` for more on the
programming model.

### Build rrun

```bash
cd cpp/build
cmake --build . --target rrun
```

### Single-Node Launch

```bash
# Launch 2 ranks on the local node
./tools/rrun -n 2 ./benchmarks/bench_comm -C ucxx -O all-to-all

# With verbose output and specific GPUs
./tools/rrun -v -n 4 -g 0,1,2,3 ./benchmarks/bench_comm -C ucxx
```
