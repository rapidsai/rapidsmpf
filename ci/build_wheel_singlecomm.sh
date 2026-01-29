#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Tests building without MPI and without UCXX. This script only ensures the build
# process succeeds even when both MPI and UCXX are disabled, in which case the
# `Single` communicator may be used.

set -euo pipefail

RAPIDS_INIT_PIP_REMOVE_NVIDIA_INDEX="true"
export RAPIDS_INIT_PIP_REMOVE_NVIDIA_INDEX
source rapids-init-pip

package_name="librapidsmpf"
package_dir="python/librapidsmpf"
package_name_py="rapidsmpf"
package_dir_py="python/rapidsmpf"

# Stage 1: build librapidsmpf
rapids-logger "Generating build requirements for ${package_name}"

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${package_name}" \
  --file-key "py_rapids_build_${package_name}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true" \
| tee /tmp/requirements-build.txt

rapids-logger "Installing build requirements for ${package_name}"
rapids-pip-retry install \
    -v \
    --prefer-binary \
    -r /tmp/requirements-build.txt

# build with '--no-build-isolation', for better sccache hit rate
# 0 really means "add --no-build-isolation" (ref: https://github.com/pypa/pip/issues/5735)
export PIP_NO_BUILD_ISOLATION=0

export SKBUILD_CMAKE_ARGS="-DBUILD_MPI_SUPPORT=OFF;-DBUILD_UCXX_SUPPORT=OFF;-DBUILD_TESTS=OFF;-DBUILD_BENCHMARKS=OFF;-DBUILD_EXAMPLES=OFF;-DBUILD_NUMA_SUPPORT=OFF"

# Needed also for librapidsmpf to find nvml.h
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export SITE_PACKAGES

./ci/build_wheel.sh "${package_name}" "${package_dir}"

python -m auditwheel repair \
    --exclude libcudf.so \
    --exclude librapids_logger.so \
    --exclude librmm.so \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    ${package_dir}/dist/*

./ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

# Stage 2: build rapidsmpf using librapidsmpf's local build
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

rapids-logger "Generating build requirements for ${package_name_py}"

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${package_name_py}" \
  --file-key "py_rapids_build_${package_name_py}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true" \
| tee /tmp/requirements-build.txt

echo "librapidsmpf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"/librapidsmpf_*.whl)" >> "${PIP_CONSTRAINT}"

rapids-logger "Installing build requirements for ${package_name_py}"
rapids-pip-retry install \
    -v \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    -r /tmp/requirements-build.txt

export SKBUILD_CMAKE_ARGS=""

./ci/build_wheel.sh "${package_name_py}" "${package_dir_py}"

python -m auditwheel repair \
    --exclude libcudf.so \
    --exclude librapids_logger.so \
    --exclude librmm.so \
    --exclude librapidsmpf.so \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    ${package_dir_py}/dist/*

# Remove librapidsmpf package before validating rapidsmpf package
rm "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"/librapidsmpf_*.whl
./ci/validate_wheel.sh "${package_dir_py}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
