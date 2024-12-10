#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail
set -x
. /opt/conda/etc/profile.d/conda.sh

RAPIDS_VERSION="$(rapids-version)"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" \
  --prepend-channel "${CPP_CHANNEL}" \
  | tee env.yaml

# Insert the two librapidsmp packages previously built by CI.
sed -i "/dependencies:/a \- librapidsmp=${RAPIDS_VERSION}" env.yaml
sed -i "/dependencies:/a \- librapidsmp-tests=${RAPIDS_VERSION}" env.yaml
# And create the conda environment.
rapids-mamba-retry env create -qy -f env.yaml -n test

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

# Trap ERR so that `EXITCODE=1` is set when a command fails
EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Run librapidsmp gtests"
./run_ctests.sh

# Ensure that benchmarks are runnable
rapids-logger "Run benchmark smoketests"

if (( ${EXITCODE} == 0 )); then
    ./run_cpp_benchmark_smoketests.sh
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
