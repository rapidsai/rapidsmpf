# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds cuCascade and ensures cuDF transitive dependencies are available for linking.
#
# NOTE: We explicitly find kvikio and nvcomp here because:
#
# 1. cuCascade depends on cuDF, which depends on kvikio and nvcomp
# 2. cuCascade runs as a CMake subdirectory/subproject and needs GLOBAL targets to see dependencies
# 3. CMake's transitive dependency handling can fail with mixed static/shared libraries
# 4. Without explicit transitive linking, the linker cannot find cuDF's dependent libraries when
#    building tools
#
# We build cuCascade as a static library to avoid packaging issues with wheels.
function(find_and_configure_cucascade)
  # Find cuDF transitive dependencies that cuCascade's subproject and the tools need.
  if(NOT TARGET kvikio::kvikio)
    find_package(kvikio REQUIRED CONFIG)
  endif()
  if(NOT TARGET nvcomp::nvcomp)
    find_package(nvcomp REQUIRED CONFIG)
  endif()

  # Mark dependencies as GLOBAL so cuCascade's subproject can see them
  if(TARGET kvikio::kvikio)
    set_target_properties(kvikio::kvikio PROPERTIES IMPORTED_GLOBAL TRUE)
  endif()
  if(TARGET nvcomp::nvcomp)
    set_target_properties(nvcomp::nvcomp PROPERTIES IMPORTED_GLOBAL TRUE)
  endif()

  rapids_cpm_find(
    cuCascade 0.1.0
    GLOBAL_TARGETS cuCascade::cucascade
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

  # Create an interface library that wraps cuCascade to avoid export conflicts. This target won't be
  # exported but can be used internally. Link cuDF transitive dependencies explicitly.
  if(TARGET cuCascade::cucascade AND NOT TARGET rapidsmpf_cucascade_internal)
    add_library(rapidsmpf_cucascade_internal INTERFACE)
    target_link_libraries(rapidsmpf_cucascade_internal INTERFACE cuCascade::cucascade)
    if(TARGET kvikio::kvikio)
      target_link_libraries(rapidsmpf_cucascade_internal INTERFACE kvikio::kvikio)
    endif()
    if(TARGET nvcomp::nvcomp)
      target_link_libraries(rapidsmpf_cucascade_internal INTERFACE nvcomp::nvcomp)
    endif()
    set_target_properties(rapidsmpf_cucascade_internal PROPERTIES EXPORT_NAME "")
  endif()
endfunction()

find_and_configure_cucascade()
