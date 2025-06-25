#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -eou pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CPP_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="librapidsmpf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYTHON_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="rapidsmpf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)

# generate test dependencies
rapids-dependency-file-generator --output requirements --file-key "test_python" --matrix "cuda=12.*;arch=x86_64;cuda_suffixed=true;py=313" | tee -a "$PIP_CONSTRAINT"

# echo to expand wildcard before adding '[extra]' requires for pip
rapids-pip-retry install \
    -v \
    --constraint "${PIP_CONSTRAINT}" \
    "${CPP_WHEELHOUSE}"/*.whl \
    "$(echo "${PYTHON_WHEELHOUSE}"/rapidsmpf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

python -m pytest -vs ./python/rapidsmpf/rapidsmpf/tests
