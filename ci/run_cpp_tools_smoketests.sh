#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -xeuo pipefail

CI_PATH="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
TIMEOUT_TOOL_PATH="${CI_PATH}"/timeout_with_stack.py
VALIDATE_TOPOLOGY_PATH="${CI_PATH}"/validate_topology_json.py

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/benchmarks/librapidsmpf/"

# Confirm no dependencies on OpenMPI variables
unset OMPI_ALLOW_RUN_AS_ROOT
unset OMPI_ALLOW_RUN_AS_ROOT_CONFIRM
unset OMPI_MCA_opal_cuda_support

python "${TIMEOUT_TOOL_PATH}" 30 \
    rrun -n 3 -g 0,0,0 ./bench_comm -m cuda -C ucxx
python "${TIMEOUT_TOOL_PATH}" 30 \
    rrun --tag-output -n 3 -g 0,0,0 ./bench_comm -m cuda -C ucxx
python "${TIMEOUT_TOOL_PATH}" 30 \
    rrun --tag-output -n 3 -g 0,0,0 ./bench_shuffle -m cuda -C ucxx
python "${TIMEOUT_TOOL_PATH}" 30 \
    rrun --tag-output -n 1 -g 0 ./bench_streaming_shuffle -m cuda -C ucxx

topology_discovery | python "${VALIDATE_TOPOLOGY_PATH}" -
