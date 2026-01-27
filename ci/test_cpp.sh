#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

CPP_CHANNEL=$(rapids-download-conda-from-github cpp)

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" \
  --prepend-channel "${CPP_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

# Support invoking test_cpp.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

# Trap ERR so that `EXITCODE` is printed when a command fails and the script
# exits with error status
EXITCODE=0
# shellcheck disable=SC2329
set_exit_code() {
    EXITCODE=$?
    rapids-logger "Test failed with exit code ${EXITCODE}"
}
trap set_exit_code ERR
set +e

rapids-logger "Run librapidsmpf gtests"
./run_ctests.sh

# Ensure that examples are runnable
rapids-logger "Run example smoketests"
./run_cpp_example_smoketests.sh

# Ensure that benchmarks are runnable
rapids-logger "Run benchmark smoketests"
./run_cpp_benchmark_smoketests.sh

# Ensure tools are runnable
rapids-logger "Run tools smoketests"
./run_cpp_tools_smoketests.sh

# Ensure rrun is runnable
rapids-logger "Run rrun gtests"
./run_rrun_tests.sh

BENCHMARKS_DIR=$CONDA_PREFIX/bin/benchmarks/librapidsmpf

rapids-logger "Validate NDSH benchmarks"
python ../cpp/scripts/validate_ndsh.py \
  --input-dir scale-1/ \
  --output-dir validation/ \
  --generate-data \
  --benchmark-dir "${BENCHMARKS_DIR}" \
  --benchmark-args='--no-pinned-host-memory'

rapids-logger "Test script exiting with exit code: $EXITCODE"
exit ${EXITCODE}
