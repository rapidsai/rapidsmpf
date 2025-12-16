# ============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# ============================================================================

# This function finds libcoro and sets any additional necessary environment variables.
function(find_and_configure_libcoro)
  if(TARGET libcoro)
    return()
  endif()

  rapids_cpm_find(
    libcoro 0.15.0
    GLOBAL_TARGETS libcoro
    BUILD_EXPORT_SET rapidsmpf-exports
    CPM_ARGS
    GIT_REPOSITORY https://github.com/jbaldwin/libcoro
    # We need a version that includes all the following PRs:
    # https://github.com/jbaldwin/libcoro/pull/399, https://github.com/jbaldwin/libcoro/pull/400,
    # https://github.com/jbaldwin/libcoro/pull/402, https://github.com/jbaldwin/libcoro/pull/423
    GIT_TAG f8dd9f744c7e5e00fa8df6aa79f8dfbd573717fd
    GIT_SHALLOW FALSE
    OPTIONS "LIBCORO_FEATURE_NETWORKING OFF"
            "LIBCORO_EXTERNAL_DEPENDENCIES OFF"
            "LIBCORO_BUILD_EXAMPLES OFF"
            "LIBCORO_FEATURE_TLS OFF"
            "LIBCORO_BUILD_TESTS OFF"
            "BUILD_SHARED_LIBS OFF"
            "CMAKE_POSITION_INDEPENDENT_CODE ON"
  )

  # Ignore compile warnings in libcoro.
  set_property(TARGET libcoro PROPERTY SYSTEM TRUE)

  # Remove old C++ flags used by libcoro, which isn't supported by TIDY.
  get_target_property(flags libcoro COMPILE_OPTIONS)
  list(FILTER flags EXCLUDE REGEX ".*-fconcepts.*|.*-fcoroutines.*")
  set_target_properties(libcoro PROPERTIES COMPILE_OPTIONS "${flags}")
  get_target_property(flags libcoro INTERFACE_COMPILE_OPTIONS)
  list(FILTER flags EXCLUDE REGEX ".*-fconcepts.*|.*-fcoroutines.*")
  set_target_properties(libcoro PROPERTIES INTERFACE_COMPILE_OPTIONS "${flags}")
endfunction()

# Save rapidsmpf's desired BUILD_SHARED_LIBS.
set(_RAPIDSMPF_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

# Find libcoro, which overwrites BUILD_SHARED_LIBS to OFF.
# <https://github.com/jbaldwin/libcoro/blob/main/CMakeLists.txt#L81>
find_and_configure_libcoro()

# Reset BUILD_SHARED_LIBS in cache if it was changed by libcoro.
if(_RAPIDSMPF_BUILD_SHARED_LIBS)
  # cmake-lint: disable=C0103
  set(BUILD_SHARED_LIBS
      ON
      CACHE INTERNAL "Reset by rapidsmpf after libcoro" FORCE
  )
endif()
