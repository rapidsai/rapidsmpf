#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
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

# Test with rrun

# Confirm no dependencies on OpenMPI variables
unset OMPI_ALLOW_RUN_AS_ROOT
unset OMPI_ALLOW_RUN_AS_ROOT_CONFIRM
unset OMPI_MCA_opal_cuda_support

python "${TIMEOUT_TOOL_PATH}" 30 \
    rrun -n 3 -g 0,0,0 ./bench_comm -m cuda -C ucxx-bootstrap
python "${TIMEOUT_TOOL_PATH}" 30 \
    rrun --tag-output -n 3 -g 0,0,0 ./bench_comm -m cuda -C ucxx-bootstrap
