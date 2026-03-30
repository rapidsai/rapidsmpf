#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -eou pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

PYTHON_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" rapidsmpf --stable --cuda "$RAPIDS_CUDA_VERSION")")
CPP_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="librapidsmpf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

TIMEOUT_TOOL_PATH="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/timeout_with_stack.py

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
    -v \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "${CPP_WHEELHOUSE}"/*.whl \
    "$(echo "${PYTHON_WHEELHOUSE}"/rapidsmpf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

python "${TIMEOUT_TOOL_PATH}" --enable-python 600 python -m pytest -v ./python/rapidsmpf/rapidsmpf/tests
