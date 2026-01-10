#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# This script runs compute-sanitizer on a single librapidsmpf test executable
# Usage: ./run_compute_sanitizer_test.sh TOOL_NAME TEST_NAME [additional gtest args...]
# Example: ./run_compute_sanitizer_test.sh memcheck single_tests
# Example: ./run_compute_sanitizer_test.sh racecheck single_tests --gtest_filter=ShufflerTest.*

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

rapids-logger "Running compute-sanitizer --tool ${TOOL_NAME} on ${TEST_NAME}"

# Support customizing the ctests' install location
# First, try the installed location (CI/conda environments)
installed_test_location="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/tests/librapidsmpf"
# Fall back to the build directory (devcontainer environments)
devcontainers_test_location="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../cpp/build/latest"

if [[ -d "${installed_test_location}" ]]; then
    TEST_DIR="${installed_test_location}"
elif [[ -d "${devcontainers_test_location}" ]]; then
    TEST_DIR="${devcontainers_test_location}"
else
    echo "Error: Test location not found. Searched:" >&2
    echo "  - ${installed_test_location}" >&2
    echo "  - ${devcontainers_test_location}" >&2
    exit 1
fi

TEST_EXECUTABLE="${TEST_DIR}/gtests/${TEST_NAME}"

if [ ! -x "${TEST_EXECUTABLE}" ]; then
  rapids-logger "Error: Test executable ${TEST_EXECUTABLE} not found or not executable"
  exit 1
fi

# Build compute-sanitizer arguments based on tool
SANITIZER_ARGS=(
  --tool "${TOOL_NAME}"
  --force-blocking-launches
  --error-exitcode=1
)

# Add tool-specific arguments
if [ "${TOOL_NAME}" = "memcheck" ]; then
  SANITIZER_ARGS+=(--track-stream-ordered-races=all)
fi

# Run compute-sanitizer on the specified test, excluding CuptiMonitorTest
compute-sanitizer \
  "${SANITIZER_ARGS[@]}" \
  "${TEST_EXECUTABLE}" \
  --gtest_filter=-CuptiMonitorTest.* \
  "$@"

EXITCODE=$?

rapids-logger "compute-sanitizer --tool ${TOOL_NAME} on ${TEST_NAME} exiting with value: $EXITCODE"
exit $EXITCODE
