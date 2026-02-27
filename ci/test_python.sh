#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-github cpp)
PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-package-name "conda_python" rapidsmpf --stable --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES}" \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" \
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

# Support invoking test_python.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

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

rapids-logger "Pytest RapidsMPF (MPI+UCXX)"
./ci/run_pytests.sh

rapids-logger "Pytest RapidsMPF (UCXX polling mode)"
RAPIDSMPF_UCXX_PROGRESS_MODE=polling ./ci/run_pytests.sh --disable-mpi

rapids-logger "Test script exiting with exit code: $EXITCODE"
exit "${EXITCODE}"
