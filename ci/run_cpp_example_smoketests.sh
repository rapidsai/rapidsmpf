#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/examples/librapidsmp/"

# Ensure that benchmarks are runnable
set -x
export OMPI_MCA_opal_cuda_support=1  # enable CUDA support in OpenMPI
mpirun -np 2 ./example_shuffle
