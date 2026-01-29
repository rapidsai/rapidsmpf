#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -eou pipefail

RAPIDS_INIT_PIP_REMOVE_NVIDIA_INDEX="true"
export RAPIDS_INIT_PIP_REMOVE_NVIDIA_INDEX
source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CPP_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="librapidsmpf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYTHON_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="rapidsmpf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)

# echo to expand wildcard before adding '[extra]' requires for pip
rapids-pip-retry install \
    -v \
    --constraint "${PIP_CONSTRAINT}" \
    "${CPP_WHEELHOUSE}"/*.whl \
    "$(echo "${PYTHON_WHEELHOUSE}"/rapidsmpf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

python -m pytest ./python/rapidsmpf/rapidsmpf/tests
