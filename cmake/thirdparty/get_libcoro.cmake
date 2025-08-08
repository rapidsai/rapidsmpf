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
    CPM_ARGS
    GIT_REPOSITORY https://github.com/madsbk/libcoro.git
    GIT_TAG rapidsmpf
    GIT_SHALLOW TRUE
    OPTIONS "LIBCORO_FEATURE_NETWORKING OFF"
            "LIBCORO_EXTERNAL_DEPENDENCIES OFF"
            "LIBCORO_BUILD_EXAMPLES OFF"
            "LIBCORO_FEATURE_TLS OFF"
            "LIBCORO_BUILD_TESTS OFF"
            "LIBCORO_BUILD_SHARED_LIBS OFF"
            "BUILD_SHARED_LIBS OFF"
  )
endfunction()

find_and_configure_libcoro()
# We have to reset `BUILD_SHARED_LIBS` since libcoro will set it to OFF.
set(BUILD_SHARED_LIBS ${RAPIDSMPF_BUILD_SHARED_LIBS})
