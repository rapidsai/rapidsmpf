#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-github cpp)

sccache --zero-stats

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version) rapids-conda-retry build \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/rapidsmpf

sccache --show-adv-stats

rapids-upload-conda-to-s3 python
