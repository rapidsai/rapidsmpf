# C++

RapidsMPF exposes a full C++ API for building high-performance distributed GPU
workloads without a Python runtime. The C++ layer is the foundation on which the
Python bindings are built.

## API Reference

```{toctree}
:maxdepth: 1

core
bootstrap
collectives
configuration
metadata-payload-exchange
shuffler
streaming
rrun
```

## Coverage

The C++ API provides access to all core RapidsMPF subsystems:

- **{doc}`Communicator <core>`**: MPI and UCXX backends for inter-process communication.
- **{doc}`Shuffler <shuffler>`**: Out-of-core, distributed payload shuffle service.
- **{doc}`Streaming Engine <streaming>`**: Asynchronous multi-GPU pipeline with Channels, Actors, and Messages.
- **{doc}`Memory <core>`**: BufferResource, spilling, pinned memory, and packed data utilities.
- **{doc}`Config <core>`**: Configuration options and environment-variable parsing.

## Shuffle Service

See {doc}`../background/shuffle-architecture` for an in-depth explanation of the
shuffle design.

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
