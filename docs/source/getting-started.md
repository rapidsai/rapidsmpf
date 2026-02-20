# Getting Started

Building rapidsmpf from source is recommended when running nightly/upstream versions,
since dependencies on non-ABI-stable libraries (e.g., pylibcudf) could cause temporary
breakage leading to issues such as segmentation faults. Stable versions can be installed
from conda or pip packages.

## Build from Source

Clone rapidsmpf and install the dependencies in a conda environment:

```bash
git clone https://github.com/rapidsai/rapidsmpf.git
cd rapidsmpf

# Choose an environment file that matches your system.
mamba env create --name rapidsmpf-dev --file conda/environments/all_cuda-131_arch-$(uname -m).yaml

# Build
./build.sh
```

### Debug Build

Debug builds can be produced by adding the `-g` flag:

```bash
./build.sh -g
```

#### AddressSanitizer-Enabled Build

Enabling the [AddressSanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizer)
is also possible with the `--asan` flag:

```bash
./build.sh -g --asan
```

C++ code built with AddressSanitizer should simply work, but there are caveats for CUDA
and Python code. Any CUDA code executing with AddressSanitizer requires
`protect_shadow_gap=0`, which can be set via an environment variable:

```bash
ASAN_OPTIONS=protect_shadow_gap=0
```

On the other hand, Python may require `LD_PRELOAD` to be set so that the AddressSanitizer
is loaded before Python. On a conda environment, for example, there is usually a
`$CONDA_PREFIX/lib/libasan.so`, and thus the application may be launched as follows:

```bash
LD_PRELOAD=$CONDA_PREFIX/lib/libasan.so python ...
```

Python applications using CUDA will require setting both environment variables described above.

## MPI

Run the test suite using MPI:

```bash
# When using OpenMPI, we need to enable CUDA support.
export OMPI_MCA_opal_cuda_support=1

# Run the suite using two MPI processes.
mpirun -np 2 cpp/build/gtests/mpi_tests

# Alternatively
cd cpp/build && ctest -R mpi_tests_2
```

We can also run the shuffle benchmark. To assign each MPI rank its own GPU, we use a
[binder script](https://github.com/LStuber/binding/blob/master/binder.sh):

```bash
# The binder script requires numactl: mamba install numactl
wget https://raw.githubusercontent.com/LStuber/binding/refs/heads/master/binder.sh
chmod a+x binder.sh
mpirun -np 2 ./binder.sh cpp/build/benchmarks/bench_shuffle
```

## UCX

The UCX test suite uses MPI for bootstrapping, so UCX tests must be launched with
`mpirun`:

```bash
# Run the suite using two processes.
mpirun -np 2 cpp/build/gtests/ucxx_tests
```

## rrun — Distributed Launcher

RapidsMPF includes `rrun`, a lightweight launcher that eliminates the MPI dependency for
multi-GPU workloads. This is particularly useful for development, testing, and
environments where MPI is not available. See the
{doc}`Streaming Engine <background/streaming-engine>` documentation for more on the
programming model.

### Single-Node Usage

```bash
# Build rrun
cd cpp/build
cmake --build . --target rrun

# Launch 2 ranks on the local node
./tools/rrun -n 2 ./benchmarks/bench_comm -C ucxx -O all-to-all

# With verbose output and specific GPUs
./tools/rrun -v -n 4 -g 0,1,2,3 ./benchmarks/bench_comm -C ucxx
```

See {doc}`cpp/index` for the full C++ rrun and multi-GPU launch guide.
