#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

package_name="rapidsmpf"
package_dir="python/rapidsmpf"

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

LIBRAPIDSMPF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp librapidsmpf rapidsmpf --cuda "$RAPIDS_CUDA_VERSION")")
echo "librapidsmpf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRAPIDSMPF_WHEELHOUSE}"/librapidsmpf_*.whl)" >> "${PIP_CONSTRAINT}"

rapids-logger "Generating build requirements"

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${package_name}" \
  --file-key "py_rapids_build_${package_name}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true" \
| tee /tmp/requirements-build.txt

rapids-logger "Installing build requirements"
rapids-pip-retry install \
    -v \
    --prefer-binary \
    -r /tmp/requirements-build.txt

# build with '--no-build-isolation', for better sccache hit rate
# 0 really means "add --no-build-isolation" (ref: https://github.com/pypa/pip/issues/5735)
export PIP_NO_BUILD_ISOLATION=0

export SKBUILD_CMAKE_ARGS=""

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export SITE_PACKAGES

# TODO: move this variable into `ci-wheel`
# Format Python limited API version string
RAPIDS_PY_API="cp${RAPIDS_PY_VERSION//./}"
export RAPIDS_PY_API

./ci/build_wheel.sh "${package_name}" "${package_dir}" --stable

python -m auditwheel repair \
    --exclude libnvidia-ml.so.1 \
    --exclude librapids_logger.so \
    --exclude librmm.so \
    --exclude libucp.so.0 \
    --exclude libucxx.so \
    --exclude libucxx_python.so \
    --exclude librapidsmpf.so \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    ${package_dir}/dist/*

./ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_python rapidsmpf rapidsmpf --stable --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
