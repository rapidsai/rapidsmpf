#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-logger "Create checks conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file-key checks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create -qy -f env.yaml -n checks
conda activate checks

rapids-logger "fetching cmake-format config"
RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"
FORMAT_FILE_URL=https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${RAPIDS_VERSION_MAJOR_MINOR}/cmake-format-rapids-cmake.json
export RAPIDS_CMAKE_FORMAT_FILE=/tmp/rapids_cmake_ci/cmake-formats-rapids-cmake.json
mkdir -p "$(dirname ${RAPIDS_CMAKE_FORMAT_FILE})"
wget -O ${RAPIDS_CMAKE_FORMAT_FILE} "${FORMAT_FILE_URL}"

# Run pre-commit checks
pre-commit run --hook-stage manual --all-files --show-diff-on-failure
