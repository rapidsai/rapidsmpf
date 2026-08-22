#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -xeuo pipefail

TIMEOUT_TOOL_PATH="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/timeout_with_stack.py

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/examples/librapidsmpf/"

# Ensure that cupti monitor example is runnable and creates the expected csv file
python "${TIMEOUT_TOOL_PATH}" 30 ./example_cupti_monitor
if [[ ! -f cupti_monitor_example.csv ]]; then
  echo "Error: cupti_monitor_example.csv was not created!"
  exit 1
fi
