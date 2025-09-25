#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -xeuo pipefail

TIMEOUT_TOOL_PATH="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/timeout_with_stack.py

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/tests/librapidsmpf/"

# OpenMPI specific options
export OMPI_ALLOW_RUN_AS_ROOT=1  # CI runs as root
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_opal_cuda_support=1  # enable CUDA support in OpenMPI

EXTRA_ARGS=("$@")

# Temporarily increasing timeouts to 5m.
# See: https://github.com/rapidsai/rapidsmpf/issues/75
timeout_secs=$((5*60)) # 5m timeout

# Run tests using mpirun with multiple nranks. Test cases and nranks are defined in the cpp/tests/CMakeLists.txt

# mpi_tests cases
python "${TIMEOUT_TOOL_PATH}" "${timeout_secs}" \
   ctest --verbose --no-tests=error --output-on-failure -R "mpi_tests_*" "${EXTRA_ARGS[@]}"

# ucxx_tests cases
python "${TIMEOUT_TOOL_PATH}" "${timeout_secs}" \
    ctest --verbose --no-tests=error --output-on-failure -R "ucxx_tests_*" "${EXTRA_ARGS[@]}"

# single_tests case
python "${TIMEOUT_TOOL_PATH}" "${timeout_secs}" \
    ctest --verbose --no-tests=error --output-on-failure  -R "single_tests" "${EXTRA_ARGS[@]}"
