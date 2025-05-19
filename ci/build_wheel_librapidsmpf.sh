#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -euo pipefail

package_name="librapidsmpf"
package_dir="python/librapidsmpf"

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

export SKBUILD_CMAKE_ARGS="-DBUILD_MPI_SUPPORT=OFF;-DBUILD_TESTS=OFF;-DBUILD_BENCHMARKS=OFF;-DBUILD_EXAMPLES=OFF"

./ci/build_wheel.sh "${package_name}" "${package_dir}"

python -m auditwheel repair \
    --exclude libcudf.so \
    --exclude librapids_logger.so \
    --exclude librmm.so \
    --exclude libucp.so.0 \
    --exclude libucxx.so \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    ${package_dir}/dist/*

./ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
