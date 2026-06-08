# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =================================================================================

# This function finds cudf_streaming for test/benchmark use only. It does NOT add cudf_streaming to
# any rapidsmpf export set.
#
# When BUILD_TESTS is enabled (in addition to BUILD_CUDF_TESTS, which gates inclusion of this file),
# the cudf `testing` component is also imported so that tests can link against `cudf::cudftestutil`
# and `cudf::cudftestutil_impl`. The default `find_dependency(cudf)` triggered transitively by
# cudf_streaming does not request the `testing` component, so we request it explicitly here.
# Benchmarks and examples only need `cudf_streaming::cudf_streaming` and do not pay for the testing
# component. This is a temporary measure until BUILD_CUDF_TESTS is removed entirely.
function(find_and_configure_cudf_streaming)

  set(oneValueArgs VERSION GIT_REPO GIT_TAG)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT TARGET cudf_streaming::cudf_streaming)
    rapids_cpm_find(
      cudf_streaming ${PKG_VERSION}
      GLOBAL_TARGETS cudf_streaming::cudf_streaming
      CPM_ARGS
      GIT_REPOSITORY ${PKG_GIT_REPO}
      GIT_TAG ${PKG_GIT_TAG}
      GIT_SHALLOW TRUE SOURCE_SUBDIR cpp/libcudf_streaming
      OPTIONS "BUILD_TESTS OFF" "BUILD_BENCHMARKS OFF" "BUILD_EXAMPLES OFF"
    )
  endif()

  # Only the cudf-dependent tests link against cudf::cudftestutil{,_impl}; skip this when tests are
  # disabled (e.g. benchmarks/examples-only builds).
  if(BUILD_TESTS AND NOT TARGET cudf::cudftestutil)
    find_package(cudf ${PKG_VERSION} REQUIRED COMPONENTS testing)
  endif()
endfunction()

find_and_configure_cudf_streaming(
  VERSION ${RAPIDS_VERSION} GIT_REPO https://github.com/rapidsai/cudf.git GIT_TAG
  "${RAPIDS_BRANCH}"
)
