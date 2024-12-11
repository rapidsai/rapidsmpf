#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/tests/librapidsmp/"

# Run gtests (single rank)
ctest --no-tests=error --output-on-failure "$@"

# Run gtests with mpirun
set -x
export OMPI_MCA_opal_cuda_support=1  # enable CUDA support in OpenMPI

# Note, we run with many different number of ranks, which we can do as long as
# the test suite only takes seconds to run.
mpirun -np 2 ctest --no-tests=error --output-on-failure "$@"
mpirun -np 3 ctest --no-tests=error --output-on-failure "$@"
mpirun -np 4 ctest --no-tests=error --output-on-failure "$@"
mpirun -np 5 ctest --no-tests=error --output-on-failure "$@"
mpirun -np 8 ctest --no-tests=error --output-on-failure "$@"
