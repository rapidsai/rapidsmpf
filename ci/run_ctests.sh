#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail
set -x

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/tests/librapidsmp/"

# Run gtests
ctest --no-tests=error --output-on-failure "$@"
