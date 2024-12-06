#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

# TODO: for now, we need to accept clobber of the fmt and librmm package.
conda config --set path_conflict warn

sccache --zero-stats

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version) rapids-conda-retry mambabuild \
    conda/recipes/librapidsmp

sccache --show-adv-stats

rapids-upload-conda-to-s3 cpp
