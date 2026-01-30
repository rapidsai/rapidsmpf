#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

RAPIDS_VERSION="$(rapids-version)"
RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"
export RAPIDS_VERSION
export RAPIDS_VERSION_MAJOR_MINOR

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

ENV_YAML_DIR="$(mktemp -d)"

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-github cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-github python)

rapids-dependency-file-generator \
  --output conda \
  --file-key docs \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n docs
conda activate docs

rapids-print-env

RAPIDS_DOCS_DIR="$(mktemp -d)"
export RAPIDS_DOCS_DIR

# Trap ERR so that `EXITCODE` is printed when a command fails and the script
# exits with error status
EXITCODE=0
# shellcheck disable=SC2329
set_exit_code() {
    EXITCODE=$?
    rapids-logger "Test failed with exit code ${EXITCODE}"
}
trap set_exit_code ERR
set +e

rapids-logger "Build CPP docs"
pushd cpp/doxygen
aws s3 cp s3://rapidsai-docs/librmm/html/"${RAPIDS_VERSION_MAJOR_MINOR}"/rmm.tag . || echo "Failed to download rmm Doxygen tag"
aws s3 cp s3://rapidsai-docs/libcudf/html/"${RAPIDS_VERSION_MAJOR_MINOR}"/libcudf.tag . || echo "Failed to download cudf Doxygen tag"

doxygen Doxyfile

mkdir -p "${RAPIDS_DOCS_DIR}/librapidsmpf/html"
mv html/* "${RAPIDS_DOCS_DIR}/librapidsmpf/html"
popd

rapids-logger "Build rapidsmpf Sphinx docs"
pushd docs/
make dirhtml O="-j 8"
mkdir -p "${RAPIDS_DOCS_DIR}/rapidsmpf/html"
mv build/dirhtml/* "${RAPIDS_DOCS_DIR}/rapidsmpf/html"
popd

RAPIDS_VERSION_NUMBER="${RAPIDS_VERSION_MAJOR_MINOR}" rapids-upload-docs

rapids-logger "Test script exiting with exit code: $EXITCODE"
exit ${EXITCODE}
