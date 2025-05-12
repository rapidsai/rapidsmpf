#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -eou pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
PYTHON_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="rapidsmpf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python ./constraints.txt

# echo to expand wildcard before adding '[extra]' requires for pip
rapids-pip-retry install \
    -v \
    --constraint ./constraints.txt \
    "$(echo "${PYTHON_WHEELHOUSE}"/rapidsmpf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"

python -m pytest ./python/rapidsmpf/rapidsmpf/tests
