# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =================================================================================

# This function finds cudf and sets any additional necessary environment variables.
function(find_and_configure_cudf)

  if(TARGET cudf::cudf)
    return()
  endif()

  set(oneValueArgs VERSION GIT_REPO GIT_TAG USE_CUDF_STATIC EXCLUDE_FROM_ALL
                   PER_THREAD_DEFAULT_STREAM
  )
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(cudf_global_targets cudf::cudf)
  set(cudf_components "")

  if(BUILD_TESTS)
    list(APPEND cudf_global_targets cudf::cudftestutil)
    set(cudf_components COMPONENTS testing)
  endif()

  rapids_cpm_find(
    cudf ${PKG_VERSION} ${cudf_components}
    GLOBAL_TARGETS ${cudf_global_targets}
    BUILD_EXPORT_SET rapidsmpf-exports
    INSTALL_EXPORT_SET rapidsmpf-exports
    CPM_ARGS
    GIT_REPOSITORY ${PKG_GIT_REPO}
    GIT_TAG ${PKG_GIT_TAG}
    GIT_SHALLOW TRUE SOURCE_SUBDIR cpp
    OPTIONS "BUILD_TESTS OFF" "BUILD_BENCHMARKS OFF" "BUILD_SHARED_LIBS ON"
            "CUDF_BUILD_TESTUTIL ${BUILD_TESTS}" "CUDF_BUILD_STREAMS_TEST_UTIL OFF"
  )

  if(TARGET cudf)
    set_property(TARGET cudf PROPERTY SYSTEM TRUE)
  endif()
endfunction()
find_and_configure_cudf(
  VERSION ${RAPIDS_VERSION} GIT_REPO https://github.com/rapidsai/cudf.git GIT_TAG "main"
)
