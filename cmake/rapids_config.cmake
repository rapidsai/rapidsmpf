# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =================================================================================

set(rapids-cmake-url
    "https://github.com/pentschev/rapids-cmake/archive/refs/heads/download-retry.zip"
)
include("../cmake/download_with_retry.cmake")

file(READ "${CMAKE_CURRENT_LIST_DIR}/../VERSION" _rapids_version)
if(_rapids_version MATCHES [[^([0-9][0-9])\.([0-9][0-9])\.([0-9][0-9])]])
  set(RAPIDS_VERSION_MAJOR "${CMAKE_MATCH_1}")
  set(RAPIDS_VERSION_MINOR "${CMAKE_MATCH_2}")
  set(RAPIDS_VERSION_PATCH "${CMAKE_MATCH_3}")
  set(RAPIDS_VERSION_MAJOR_MINOR "${RAPIDS_VERSION_MAJOR}.${RAPIDS_VERSION_MINOR}")
  set(RAPIDS_VERSION "${RAPIDS_VERSION_MAJOR}.${RAPIDS_VERSION_MINOR}.${RAPIDS_VERSION_PATCH}")
else()
  string(REPLACE "\n" "\n  " _rapids_version_formatted "  ${_rapids_version}")
  message(
    FATAL_ERROR
      "Could not determine RAPIDS version. Contents of VERSION file:\n${_rapids_version_formatted}"
  )
endif()

if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/RAPIDSMP_RAPIDS-${RAPIDS_VERSION_MAJOR_MINOR}.cmake")
  rapids_download_with_retry(
    "https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${RAPIDS_VERSION_MAJOR_MINOR}/RAPIDS.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/RAPIDSMP_RAPIDS-${RAPIDS_VERSION_MAJOR_MINOR}.cmake"
  )
endif()
include("${CMAKE_CURRENT_BINARY_DIR}/RAPIDSMP_RAPIDS-${RAPIDS_VERSION_MAJOR_MINOR}.cmake")
