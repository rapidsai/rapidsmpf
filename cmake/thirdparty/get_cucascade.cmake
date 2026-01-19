# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

# This function finds cuCascade and sets any additional necessary environment variables.
#
# NOTE: We explicitly find RMM, cuDF, and KvikIO targets here even though they are already
# configured earlier in the build process. This is necessary because:
#
# 1. cuCascade runs as a CMake subdirectory/subproject and needs GLOBAL targets to see them
# 2. We need to extract rmm_ROOT and cudf_ROOT from the targets to ensure cuCascade uses the same
#    RMM/cuDF instances as rapidsmpf
# 3. Prevents cuCascade from building its own copies of RMM/cuDF via CPM
# 4. KvikIO is a dependency of cuDF and must be explicitly found and linked to ensure proper
#    transitive linking (cuDF's CMake config may not properly propagate KvikIO dependencies)
function(find_and_configure_cucascade)
  # Ensure rmm, cudf, and kvikio are found first
  if(NOT TARGET rmm::rmm)
    find_package(rmm REQUIRED CONFIG)
  endif()
  if(NOT TARGET cudf::cudf)
    find_package(cudf REQUIRED CONFIG)
  endif()
  if(NOT TARGET kvikio::kvikio)
    find_package(kvikio REQUIRED CONFIG)
  endif()

  set(cucascade_version "1.0.0")
  set(cucascade_fork "pentschev")
  set(cucascade_pinned_tag "update-rmm")

  # Mark targets as GLOBAL so they're visible to cuCascade
  foreach(_target rmm::rmm cudf::cudf kvikio::kvikio CUDA::cudart CUDA::nvml)
    if(TARGET ${_target})
      set_target_properties(${_target} PROPERTIES IMPORTED_GLOBAL TRUE)
    endif()
  endforeach()

  # Get the actual package directories to ensure cuCascade uses the same RMM/cuDF
  get_target_property(_rmm_location rmm::rmm LOCATION)
  if(_rmm_location)
    get_filename_component(_rmm_lib_dir "${_rmm_location}" DIRECTORY)
    get_filename_component(rmm_ROOT "${_rmm_lib_dir}/.." ABSOLUTE)
  endif()

  get_target_property(_cudf_location cudf::cudf LOCATION)
  if(_cudf_location)
    get_filename_component(_cudf_lib_dir "${_cudf_location}" DIRECTORY)
    get_filename_component(cudf_ROOT "${_cudf_lib_dir}/.." ABSOLUTE)
  endif()

  message(STATUS "RMM root directory for cuCascade: ${rmm_ROOT}")
  message(STATUS "cuDF root directory for cuCascade: ${cudf_ROOT}")

  # Use rapids_cpm_find and force cuCascade to use the same RMM/cuDF
  set(ENV{rmm_ROOT} "${rmm_ROOT}")
  set(ENV{cudf_ROOT} "${cudf_ROOT}")

  rapids_cpm_find(
    cuCascade ${cucascade_version}
    GLOBAL_TARGETS cuCascade::cucascade
    CPM_ARGS
    GIT_REPOSITORY https://github.com/${cucascade_fork}/cuCascade.git
    GIT_TAG ${cucascade_pinned_tag}
    GIT_SHALLOW TRUE
    OPTIONS "BUILD_TESTS OFF" "BUILD_SHARED_LIBS ON" "BUILD_STATIC_LIBS OFF"
            "WARNINGS_AS_ERRORS OFF" "rmm_ROOT ${rmm_ROOT}" "cudf_ROOT ${cudf_ROOT}"
    EXCLUDE_FROM_ALL
  )

  # Create an interface library that wraps cuCascade to avoid export conflicts This target won't be
  # exported but can be used internally Also link KvikIO to ensure transitive dependencies are
  # satisfied (cuDF depends on KvikIO)
  if(TARGET cuCascade::cucascade AND NOT TARGET rapidsmpf_cucascade_internal)
    add_library(rapidsmpf_cucascade_internal INTERFACE)
    target_link_libraries(rapidsmpf_cucascade_internal INTERFACE cuCascade::cucascade)
    # Link KvikIO to ensure cuDF's dependency is satisfied
    if(TARGET kvikio::kvikio)
      target_link_libraries(rapidsmpf_cucascade_internal INTERFACE kvikio::kvikio)
    endif()
    # Mark this as not exported
    set_target_properties(rapidsmpf_cucascade_internal PROPERTIES EXPORT_NAME "")
  endif()

  # Set up installation of cuCascade library for wheel packaging Since cuCascade is built with
  # EXCLUDE_FROM_ALL, we need to explicitly install the library files This install CODE runs during
  # 'cmake --install' when the library files exist
  if(TARGET cuCascade::cucascade)
    # Get the install library directory (will be set by rapids_cmake_install_lib_dir in main
    # CMakeLists) We'll use a relative path that will be resolved at install time
    set(_cucascade_build_dir "${CMAKE_BINARY_DIR}/_deps/cucascade-build")

    # Install CODE that finds and copies cuCascade library files This must be called from the main
    # CMakeLists.txt after rapids_cmake_install_lib_dir is called Store the build directory in a
    # cache variable for use in install CODE
    set(_CUCASCADE_BUILD_DIR
        "${_cucascade_build_dir}"
        CACHE INTERNAL "cuCascade build directory"
    )
  endif()
endfunction()

find_and_configure_cucascade()
