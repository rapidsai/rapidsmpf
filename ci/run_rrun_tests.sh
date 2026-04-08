#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -xeuo pipefail

TIMEOUT_TOOL_PATH="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/timeout_with_stack.py

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/tests/librapidsmpf/"

EXTRA_ARGS=("$@")

# CI timeout
timeout_secs=15

# rrun_tests case, these need to run with rrun and not ctest/mpirun.
for nrank in 1 2 3 4 5 8; do
  python "${TIMEOUT_TOOL_PATH}" "${timeout_secs}" \
      rrun -n "${nrank}" --bind-to all ./gtests/rrun_tests "${EXTRA_ARGS[@]}"
done

# rrun tests should also work when not running with `rrun` CLI. E.g., resource bindings
# need to work outside of `rrun`, which is the intended use case for
# `rapidsmpf::rrun::bind()`.
./gtests/rrun_tests "${EXTRA_ARGS[@]}"
