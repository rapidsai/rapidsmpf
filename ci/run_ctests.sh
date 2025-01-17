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
    mpirun -np "$nrank" ctest --no-tests=error --output-on-failure \
        -R "$test" $EXTRA_ARGS
}

for nrank in (2 3 4 5 8); do
    run_mpirun_test $nrank mpi_tests
done


# run_mpirun_test 1 mpi_tests
# run_mpirun_test 2 mpi_tests
# run_mpirun_test 4 mpi_tests
# run_mpirun_test 4 mpi_tests
# run_mpirun_test 4 mpi_tests

# # Run gtests (single rank)
# ctest --no-tests=error --output-on-failure "$@"

# # Run gtests with mpirun. Note, we run with many different number of ranks,
# # which we can do as long as the test suite only takes seconds to run.
# mpirun -np 2 ctest --no-tests=error --output-on-failure "$@"
# mpirun -np 3 ctest --no-tests=error --output-on-failure "$@"
# mpirun -np 4 ctest --no-tests=error --output-on-failure "$@"
# mpirun -np 5 ctest --no-tests=error --output-on-failure "$@"
# mpirun -np 8 ctest --no-tests=error --output-on-failure "$@"
