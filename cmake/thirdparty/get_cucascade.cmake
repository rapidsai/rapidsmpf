# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds cuCascade and ensures kvikio is available for transitive linking.
#
# NOTE: We explicitly find kvikio here because:
#
# 1. cuCascade depends on cuDF, which depends on kvikio
# 2. cuCascade runs as a CMake subdirectory/subproject and needs GLOBAL targets to see dependencies
# 3. CMake's transitive dependency handling can fail with mixed static/shared libraries
# 4. Without explicit kvikio linking, the linker cannot find libkvikio.so when building tools
#
# We build cuCascade as a static library to avoid packaging issues with wheels.
function(find_and_configure_cucascade)
  # Find kvikio to satisfy cuDF's dependency chain
  if(NOT TARGET kvikio::kvikio)
    find_package(kvikio REQUIRED CONFIG)
  endif()

  # Mark kvikio as GLOBAL so cuCascade's subproject can see it
  if(TARGET kvikio::kvikio)
    set_target_properties(kvikio::kvikio PROPERTIES IMPORTED_GLOBAL TRUE)
  endif()

  rapids_cpm_find(
    cuCascade 0.1.0
    GLOBAL_TARGETS cuCascade::cucascade
    BUILD_EXPORT_SET rapidsmpf-exports
    CPM_ARGS
    GIT_REPOSITORY https://github.com/NVIDIA/cuCascade.git
    GIT_TAG main
    GIT_SHALLOW TRUE
    OPTIONS "CUCASCADE_BUILD_TESTS OFF"
            "CUCASCADE_BUILD_BENCHMARKS OFF"
            "CUCASCADE_BUILD_SHARED_LIBS OFF"
            "CUCASCADE_BUILD_STATIC_LIBS ON"
            "CUCASCADE_WARNINGS_AS_ERRORS OFF"
    EXCLUDE_FROM_ALL
  )
endfunction()

find_and_configure_cucascade()
