#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking run_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/rapidsmp/rapidsmp

# OpenMPI specific options
export OMPI_ALLOW_RUN_AS_ROOT=1  # CI runs as root
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_opal_cuda_support=1  # enable CUDA support in OpenMPI

# Run tests (single rank)
pytest --cache-clear --verbose "$@" tests

# Run tests with mpirun. Note, we run with many different number of ranks,
# which we can do as long as the test suite only takes seconds to run.
mpirun -np 2 python -m pytest --cache-clear --verbose "$@" tests
mpirun -np 3 python -m pytest --cache-clear --verbose "$@" tests
mpirun -np 4 python -m pytest --cache-clear --verbose "$@" tests
mpirun -np 5 python -m pytest --cache-clear --verbose "$@" tests
mpirun -np 8 python -m pytest --cache-clear --verbose "$@" tests
