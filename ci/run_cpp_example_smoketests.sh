#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/examples/librapidsmp/"

# Ensure that benchmarks are runnable
OMPI_MCA_opal_cuda_support=1 mpirun --allow-run-as-root -np 2 ./bench_shuffle
