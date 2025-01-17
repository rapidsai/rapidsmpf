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
    local nrank="$1"  # Number of ranks
    local test="$2"   # Test name
    echo "Running test: $test with $nrank ranks"

    # Run the test with timeout and check for errors
    if ! timeout 1m mpirun -np "$nrank" ctest --no-tests=error --output-on-failure \
        -R "$test" $EXTRA_ARGS; then
        local exit_code=$?
        if [ "$exit_code" -eq 124 ]; then
            echo "Error: Test $test with $nrank ranks timed out." >&2
        fi
        exit $exit_code  # Exit the script with the same error code
    fi
}

for nrank in 2 3 4 5 8; do
    run_mpirun_test $nrank mpi_tests
done

for nrank in 2 3 4 5 8; do
    run_mpirun_test $nrank ucxx_tests
done
