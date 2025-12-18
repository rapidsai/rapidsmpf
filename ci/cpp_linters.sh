#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-logger "Create checks conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-dependency-file-generator \
  --output conda \
  --file-key clang_tidy \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create -qy -f env.yaml -n clang_tidy

# Temporarily allow unbound variables for conda activation.
set +u
conda activate clang_tidy
set -u

source rapids-configure-sccache

# Run the build via CMake, which will run clang-tidy when RAPIDSMPF_CLANG_TIDY is enabled.
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DRAPIDSMPF_CLANG_TIDY=ON -DBUILD_CUPTI_SUPPORT=ON -DCMAKE_CUDA_ARCHITECTURES=75 -GNinja
cmake --build cpp/build
