#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

# This script sets up the test environment and runs compute-sanitizer on librapidsmpf tests
# Usage: ./test_cpp_sanitizer.sh TOOL_NAME TEST_NAME [additional gtest args...]
# Example: ./test_cpp_sanitizer.sh memcheck single_tests
# Example: ./test_cpp_sanitizer.sh racecheck single_tests --gtest_filter=ShufflerTest.*

if [ $# -lt 2 ]; then
  echo "Error: Tool and test name required"
  echo "Usage: $0 TOOL_NAME TEST_NAME [additional gtest args...]"
  echo "  TOOL_NAME: compute-sanitizer tool (memcheck, racecheck, initcheck, synccheck)"
  echo "  TEST_NAME: librapidsmpf test name (e.g., single_tests)"
  exit 1
fi

TOOL_NAME="${1}"
shift
TEST_NAME="${1}"
shift

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

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

# Support invoking test_cpp_sanitizer.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

./run_compute_sanitizer_test.sh "${TOOL_NAME}" "${TEST_NAME}" "$@"
