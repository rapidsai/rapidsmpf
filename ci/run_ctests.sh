#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -xeuo pipefail

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/tests/librapidsmp/"

# OpenMPI specific options
export OMPI_ALLOW_RUN_AS_ROOT=1  # CI runs as root
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_opal_cuda_support=1  # enable CUDA support in OpenMPI

EXTRA_ARGS="$@"
run_mpirun_test() {
    local timeout="$1" # Timeout
    local nrank="$2"   # Number of ranks
    echo "Running ctest with $nrank ranks"
    timeout "$timeout" mpirun -np "$nrank" ctest --no-tests=error \
        --output-on-failure $EXTRA_ARGS
}

# Note, we run with many different number of ranks, which we can do as long as
# the test suite only takes seconds to run (timeouts after one minute).
for nrank in 1 2 3 4 5 8; do
    run_mpirun_test 1m $nrank
done
