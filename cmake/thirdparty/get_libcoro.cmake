# ============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
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
    # Pick a version that incl. <https://github.com/jbaldwin/libcoro/pull/371>
    GIT_TAG 1d472a8e91e8058d637d0bfbba1d5df93fefc3f0
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
