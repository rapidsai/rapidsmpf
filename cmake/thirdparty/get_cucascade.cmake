# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Helper to get include directory from target or fallback
function(_get_include_dir_from_target target_name out_var)
  get_target_property(_includes ${target_name} INTERFACE_INCLUDE_DIRECTORIES)
  if(_includes)
    list(GET _includes 0 _result)
  elseif(DEFINED ENV{CONDA_PREFIX})
    set(_result "$ENV{CONDA_PREFIX}/include")
  else()
    set(_result "${CMAKE_PREFIX_PATH}/include")
  endif()
  set(${out_var}
      "${_result}"
      PARENT_SCOPE
  )
endfunction()

# Finds and configure cuCascade as static library. Using it as static library avoids packaging
# issues with wheels.
function(find_and_configure_cucascade)
  rapids_cpm_find(
    cuCascade 0.1.0
    GLOBAL_TARGETS cuCascade::cucascade
    CPM_ARGS
    GIT_REPOSITORY https://github.com/NVIDIA/cuCascade.git
    GIT_TAG main
    GIT_SHALLOW TRUE
    OPTIONS "BUILD_TESTS OFF" "BUILD_BENCHMARKS OFF" "BUILD_SHARED_LIBS OFF" "BUILD_STATIC_LIBS ON"
            "WARNINGS_AS_ERRORS OFF"
    EXCLUDE_FROM_ALL
  )

  # Create an interface library that wraps cuCascade to avoid export conflicts This target won't be
  # exported but can be used internally.
  if(TARGET cuCascade::cucascade AND NOT TARGET rapidsmpf_cucascade_internal)
    add_library(rapidsmpf_cucascade_internal INTERFACE)
    target_link_libraries(rapidsmpf_cucascade_internal INTERFACE cuCascade::cucascade)
    # Link KvikIO to ensure cuDF's dependency is satisfied
    set_target_properties(rapidsmpf_cucascade_internal PROPERTIES EXPORT_NAME "")
  endif()
endfunction()

find_and_configure_cucascade()
