#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -xeuo pipefail

TIMEOUT_TOOL_PATH="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/timeout_with_stack.py

# Support customizing the ctests' install location
# First, try the installed location (CI/conda environments)
installed_test_location="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/tests/librapidsmpf/"
# Fall back to the build directory (devcontainer environments)
devcontainers_test_location="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../cpp/build/latest"

if [[ -d "${installed_test_location}" ]]; then
    cd "${installed_test_location}"
elif [[ -d "${devcontainers_test_location}" ]]; then
    cd "${devcontainers_test_location}"
else
    echo "Error: Test location not found. Searched:" >&2
    echo "  - ${installed_test_location}" >&2
    echo "  - ${devcontainers_test_location}" >&2
    exit 1
fi

# OpenMPI specific options
export OMPI_ALLOW_RUN_AS_ROOT=1  # CI runs as root
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_opal_cuda_support=1  # enable CUDA support in OpenMPI

EXTRA_ARGS=("$@")

# CI timeout
timeout_secs=$((10*60)) # 10m

# Run tests using mpirun with multiple nranks. Test cases and nranks are defined in the cpp/tests/CMakeLists.txt
# Each nrank count gets its own timeout budget to prevent slow high-rank tests from being starved
# by the cumulative runtime of earlier tests.
nranks_to_run=(1 2 3 4 5 8)

# mpi_tests cases
for nranks in "${nranks_to_run[@]}"; do
    python "${TIMEOUT_TOOL_PATH}" "${timeout_secs}" \
        ctest --verbose --no-tests=error --output-on-failure -R "^mpi_tests_${nranks}$" "${EXTRA_ARGS[@]}"
done

# ucxx_tests cases, includes both default (thread-blocking) and polling progress modes
for nranks in "${nranks_to_run[@]}"; do
    python "${TIMEOUT_TOOL_PATH}" "${timeout_secs}" \
        ctest --verbose --no-tests=error --output-on-failure -R "^ucxx_tests_${nranks}$" "${EXTRA_ARGS[@]}"
    RAPIDSMPF_UCXX_PROGRESS_MODE=polling python "${TIMEOUT_TOOL_PATH}" "${timeout_secs}" \
        ctest --verbose --no-tests=error --output-on-failure -R "^ucxx_tests_${nranks}$" "${EXTRA_ARGS[@]}"
done

# single_tests case
python "${TIMEOUT_TOOL_PATH}" "${timeout_secs}" \
    ctest --verbose --no-tests=error --output-on-failure  -R "single_tests" "${EXTRA_ARGS[@]}"
