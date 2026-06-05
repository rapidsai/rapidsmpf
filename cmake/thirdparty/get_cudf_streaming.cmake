# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =================================================================================

# This function finds cudf_streaming for test/benchmark use only.
# It does NOT add cudf_streaming to any rapidsmpf export set.
function(find_and_configure_cudf_streaming)

  if(TARGET cudf_streaming::cudf_streaming)
    return()
  endif()

  set(oneValueArgs VERSION GIT_REPO GIT_TAG)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  rapids_cpm_find(
    cudf_streaming ${PKG_VERSION}
    GLOBAL_TARGETS cudf_streaming::cudf_streaming
    CPM_ARGS
    GIT_REPOSITORY ${PKG_GIT_REPO}
    GIT_TAG ${PKG_GIT_TAG}
    GIT_SHALLOW TRUE SOURCE_SUBDIR cpp/libcudf_streaming
    OPTIONS "BUILD_TESTS OFF" "BUILD_BENCHMARKS OFF" "BUILD_EXAMPLES OFF"
  )
endfunction()

find_and_configure_cudf_streaming(
  VERSION ${RAPIDS_VERSION} GIT_REPO https://github.com/rapidsai/cudf.git GIT_TAG
  "${RAPIDS_BRANCH}"
)
