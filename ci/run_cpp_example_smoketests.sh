#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -xeuo pipefail

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/examples/librapidsmpf/"

# OpenMPI specific options
export OMPI_ALLOW_RUN_AS_ROOT=1  # CI runs as root
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_opal_cuda_support=1  # enable CUDA support in OpenMPI

# Ensure that shuffle example is runnable
mpirun --map-by node --bind-to none -np 2 ./example_shuffle

# Ensure that cupti monitor example is runnable and creates the expected csv file
./example_cupti_monitor
if [[ ! -f cupti_monitor_example.csv ]]; then
  echo "Error: cupti_monitor_example.csv was not created!"
  exit 1
fi
