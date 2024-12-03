# RAPIDS-MP

Collection of multi-gpu, distributed memory algorithms.

## Getting started

Currently, there is no conda or pip packages for rapidsmp thus we have to build from source.

Clone rapidsmp and install the dependencies in a conda environment:
```
git clone https://github.com/rapidsai/rapids-multi-gpu.git
cd rapids-multi-gpu

# Choose a environment file that match your system.
mamba env create --name rapidsmp-dev --file conda/environments/all_cuda-125_arch-x86_64.yaml

# Build
cd cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Run the test suite using MPI:
```
# When using OpenMP, we need to enable CUDA support.
export OMPI_MCA_opal_cuda_support=1

# Run the suite using two MPI processes.
mpirun -np 2 build/gtests/mpi_tests
```

We can also run the shuffle benchmark. To assign each MPI rank its own GPU, we use a [binder script](https://github.com/LStuber/binding/blob/master/binder.sh):
```
# The binder script requires numactl `mamba install numactl`
wget https://raw.githubusercontent.com/LStuber/binding/refs/heads/master/binder.sh
chmod a+x binder.sh
mpirun -np 2 ./binder.sh build/benchmarks/bench_shuffle
```

## Algorithms
### Table Shuffle Service
Example of a MPI program that uses the shuffler:

https://github.com/madsbk/rapids-multi-gpu/blob/6026d00a4262299e8f2e98fdf0e7010f6da67198/cpp/examples/example_shuffle.cpp

## Communicator

### MPI

### UCX
