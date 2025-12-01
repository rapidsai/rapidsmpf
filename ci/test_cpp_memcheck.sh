#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -xeuo pipefail

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

# Trap ERR so that `EXITCODE` is printed when a command fails and the script
# exits with error status
EXITCODE=0
# shellcheck disable=SC2317
set_exit_code() {
    EXITCODE=$?
    rapids-logger "Test failed with exit code ${EXITCODE}"
}
trap set_exit_code ERR
set +e

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/tests/librapidsmpf/"

rapids-logger "Run librapidsmpf gtests with compute-sanitizer (Single Node)"
compute-sanitizer --tool memcheck --track-stream-ordered-races=all gtests/single_tests --gtest_filter=-CuptiMonitorTest.*

rapids-logger "Test script exiting with exit code: $EXITCODE"
exit ${EXITCODE}
