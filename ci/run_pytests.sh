#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -xeuo pipefail

# Support invoking run_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/rapidsmp/rapidsmp

# OpenMPI specific options
export OMPI_ALLOW_RUN_AS_ROOT=1  # CI runs as root
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_opal_cuda_support=1  # enable CUDA support in OpenMPI

EXTRA_ARGS="$@"
run_mpirun_test() {
    local timeout="$1" # Timeout
    local nrank="$2"   # Number of ranks
    echo "Running pytest with $nrank ranks"
    timeout "$timeout" mpirun --map-by node --bind-to none -np "$nrank" \
        python -m pytest --cache-clear --verbose $EXTRA_ARGS tests
}

# Note, we run with many different number of ranks, which we can do as long as
# the test suite only takes seconds to run (timeouts after one minute).
for nrank in 1 2 3 4 5 8; do
    run_mpirun_test 2m $nrank
done
