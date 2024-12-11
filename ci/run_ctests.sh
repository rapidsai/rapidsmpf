#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -xeuo pipefail

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/tests/librapidsmp/"

# OpenMPI specific options
export OMPI_ALLOW_RUN_AS_ROOT=1  # CI runs as root
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_opal_cuda_support=1  # enable CUDA support in OpenMPI

# Run gtests (single rank)
ctest --no-tests=error --output-on-failure "$@"

# Run gtests with mpirun. Note, we run with many different number of ranks,
# which we can do as long as the test suite only takes seconds to run.
mpirun -np 2 ctest --no-tests=error --output-on-failure "$@"
mpirun -np 3 ctest --no-tests=error --output-on-failure "$@"
mpirun -np 4 ctest --no-tests=error --output-on-failure "$@"
mpirun -np 5 ctest --no-tests=error --output-on-failure "$@"
mpirun -np 8 ctest --no-tests=error --output-on-failure "$@"
