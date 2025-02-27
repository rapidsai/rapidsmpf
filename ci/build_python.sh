#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

sccache --zero-stats

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version) rapids-conda-retry build \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/rapidsmp

sccache --show-adv-stats

rapids-upload-conda-to-s3 python
