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

We can also run the shuffle benchmark using MPI. However, to assign each MPI rank its own GPU, we set `CUDA_VISIBLE_DEVICES`:
```
mpirun -np 1 -x CUDA_VISIBLE_DEVICES=0 build/benchmarks/bench_shuffle : -np 1 -x CUDA_VISIBLE_DEVICES=1
```

## Algorithms
### Table Shuffle Service

## Communicator

### MPI

### UCX
