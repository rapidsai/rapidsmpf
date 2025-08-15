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
            "BUILD_SHARED_LIBS OFF"
            "CMAKE_POSITION_INDEPENDENT_CODE ON"
    EXCLUDE_FROM_ALL YES # Don't install liblibcoro.a (not a typo), it is only needed when building
                         # librapidsmpf.so
  )
endfunction()

find_and_configure_libcoro()
