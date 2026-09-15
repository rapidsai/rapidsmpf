#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -xeuo pipefail

TIMEOUT_TOOL_PATH="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/timeout_with_stack.py

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/benchmarks/librapidsmpf/"

# OpenMPI specific options
export OMPI_ALLOW_RUN_AS_ROOT=1  # CI runs as root
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_opal_cuda_support=1  # enable CUDA support in OpenMPI

# Ensure that benchmarks are runnable
python "${TIMEOUT_TOOL_PATH}" 30 \
    mpirun --map-by node --bind-to none -np 3 ./bench_comm -m cuda

python "${TIMEOUT_TOOL_PATH}" 30 \
    mpirun --map-by node --bind-to none -np 3 ./bench_shuffle -m cuda -n 65536 -o 1

DISK_SPILL_DIR="$(mktemp -d)"
trap 'rm -rf "${DISK_SPILL_DIR}"' EXIT
python "${TIMEOUT_TOOL_PATH}" 30 \
    env RAPIDSMPF_SPILL_DEVICE_LIMIT=0B \
        RAPIDSMPF_SPILL_HOST_LIMIT=0B \
        RAPIDSMPF_PINNED_MEMORY=false \
        RAPIDSMPF_PERIODIC_SPILL_CHECK=disabled \
        RAPIDSMPF_DISK_SPILL_DIR="${DISK_SPILL_DIR}" \
    mpirun --map-by node --bind-to none -np 1 ./bench_shuffle -m cuda -n 4096 -o 2 -p 2
if [[ -n "$(find "${DISK_SPILL_DIR}" -type f -print -quit)" ]]; then
  echo "Error: bench_shuffle left spill files under ${DISK_SPILL_DIR}"
  exit 1
fi

RAPIDSMPF_SMOKE_TEST_MODE="ON" \
    python "${TIMEOUT_TOOL_PATH}" 30 ./bench_memory_resources

# Ensure that comm benchmark with CUPTI monitor is runnable and creates the expected csv files
python "${TIMEOUT_TOOL_PATH}" 30 \
    mpirun --map-by node --bind-to none -np 3 ./bench_comm -m cuda -M cupti_comm
for i in {0..2}; do
  if [[ ! -f cupti_comm${i}.csv ]]; then
    echo "Error: cupti_comm${i}.csv was not created!"
    exit 1
  fi
done
