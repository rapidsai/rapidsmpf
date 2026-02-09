#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -eou pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

PYTHON_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" rapidsmpf --stable --cuda "$RAPIDS_CUDA_VERSION")")
CPP_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="librapidsmpf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)

# echo to expand wildcard before adding '[extra]' requires for pip
rapids-pip-retry install \
    -v \
    --constraint "${PIP_CONSTRAINT}" \
    "${CPP_WHEELHOUSE}"/*.whl \
    "$(echo "${PYTHON_WHEELHOUSE}"/rapidsmpf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

python -m pytest ./python/rapidsmpf/rapidsmpf/tests
