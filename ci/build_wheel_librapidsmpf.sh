#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

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

export SKBUILD_CMAKE_ARGS="-DBUILD_MPI_SUPPORT=OFF;-DBUILD_TESTS=OFF;-DBUILD_BENCHMARKS=OFF;-DBUILD_EXAMPLES=OFF;-DBUILD_NUMA_SUPPORT=OFF;-DRAPIDSMPF_CLANG_TIDY=OFF"

# Needed to find nvml.h
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export SITE_PACKAGES

./ci/build_wheel.sh "${package_name}" "${package_dir}"

# patchelf (used internally by auditwheel) corrupts ELF executables when it
# must grow PT_LOAD segments to fit new RPATH data.  Shared libraries are fine,
# but standalone binaries like rrun end up with .dynstr/.dynamic outside any
# loadable segment, causing a dynamic-linker crash before main().
# The -Wl,-z,noseparate-code linker flag mitigates this on most toolchains but
# is not universally sufficient, so we also save the original binary before
# auditwheel runs and re-inject it afterward.  The _rrun.py entry-point shim
# sets LD_LIBRARY_PATH so the restored binary finds the excluded RAPIDS libs.
UNREPAIRED_WHEEL=$(ls "${package_dir}"/dist/*.whl)
RRUN_ORIG=$(mktemp)
python3 - "${UNREPAIRED_WHEEL}" "${RRUN_ORIG}" <<'EOF'
import shutil, sys, zipfile
whl, dst = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(whl) as z, z.open("librapidsmpf/bin/rrun") as src, open(dst, "wb") as f:
    shutil.copyfileobj(src, f)
EOF
chmod 755 "${RRUN_ORIG}"

python -m auditwheel repair \
    --exclude libcudf.so \
    --exclude libkvikio.so \
    --exclude libnvidia-ml.so.1 \
    --exclude librapids_logger.so \
    --exclude librmm.so \
    --exclude libucp.so.0 \
    --exclude libucxx.so \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    "${UNREPAIRED_WHEEL}"

REPAIRED_WHEEL=$(ls "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"/*.whl)
python3 - "${REPAIRED_WHEEL}" "${RRUN_ORIG}" <<'EOF'
import os, sys, zipfile
whl, rrun = sys.argv[1], sys.argv[2]
tmp = whl + ".tmp"
with zipfile.ZipFile(whl) as zin, zipfile.ZipFile(tmp, "w") as zout:
    for item in zin.infolist():
        data = open(rrun, "rb").read() if item.filename == "librapidsmpf/bin/rrun" else zin.read(item.filename)
        zout.writestr(item, data)
os.replace(tmp, whl)
EOF

./ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
