#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
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
    mpirun --map-by node --bind-to none -np 3 ./bench_shuffle -m cuda

python "${TIMEOUT_TOOL_PATH}" 30 \
    mpirun --map-by node --bind-to none -np 3 ./bench_comm -m cuda

python "${TIMEOUT_TOOL_PATH}" 30 \
  ./bench_memory_resources --benchmark_min_time=0s

python "${TIMEOUT_TOOL_PATH}" 30 \
  ./bench_streaming_shuffle -m cuda

# Ensure that shuffle benchmark with CUPTI monitor is runnable and creates the expected csv files
python "${TIMEOUT_TOOL_PATH}" 30 \
    mpirun --map-by node --bind-to none -np 3 ./bench_shuffle -m cuda -M cupti_shuffle
for i in {0..2}; do
  if [[ ! -f cupti_shuffle${i}.csv ]]; then
    echo "Error: cupti_shuffle${i}.csv was not created!"
    exit 1
  fi
done

# Ensure that comm benchmark with CUPTI monitor is runnable and creates the expected csv files
python "${TIMEOUT_TOOL_PATH}" 30 \
    mpirun --map-by node --bind-to none -np 3 ./bench_comm -m cuda -M cupti_comm
for i in {0..2}; do
  if [[ ! -f cupti_comm${i}.csv ]]; then
    echo "Error: cupti_comm${i}.csv was not created!"
    exit 1
  fi
done

# bench pack smoketest (only run 1MB buffer benchmarks)
python "${TIMEOUT_TOOL_PATH}" 30 ./bench_pack --benchmark_filter="/1/" --benchmark_min_time=0s
