#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version)
export RAPIDS_PACKAGE_VERSION

# populates `RATTLER_CHANNELS` array and `RATTLER_ARGS` array
source rapids-rattler-channel-string

# Builds with:
# --no-build-id allows for caching with `sccache`
# --channel-priority strict
rattler-build build --recipe conda/recipes/librapidsmpf \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

sccache --show-adv-stats

# remove build_cache directory to avoid uploading the entire source tree
# tracked in https://github.com/prefix-dev/rattler-build/issues/1424
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache
