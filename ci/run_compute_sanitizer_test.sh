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

# Navigate to test installation directory
TEST_DIR="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/tests/librapidsmpf"
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
