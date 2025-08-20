# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =================================================================================

# Find CUPTI (CUDA Profiling Tools Interface)

# Find CUPTI headers and library
find_path(
  CUPTI_INCLUDE_DIR
  NAMES cupti.h
  HINTS ${CUDAToolkit_INCLUDE_DIRS}
  PATHS /usr/local/cuda/include
        /usr/local/cuda/extras/CUPTI/include
        ${CUDA_TOOLKIT_ROOT_DIR}/include
        ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include
        ${CUDAToolkit_TARGET_DIR}/include
        ${CUDAToolkit_TARGET_DIR}/extras/CUPTI/include
  DOC "Path to cupti.h"
)

find_library(
  CUPTI_LIBRARY
  NAMES cupti
  HINTS ${CUDAToolkit_LIBRARY_DIR}
  PATHS /usr/local/cuda/lib64
        /usr/local/cuda/extras/CUPTI/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64
        ${CUDAToolkit_TARGET_DIR}/lib64
        ${CUDAToolkit_TARGET_DIR}/extras/CUPTI/lib64
  DOC "Path to CUPTI library"
)

# Handle the QUIETLY and REQUIRED arguments and set CUPTI_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUPTI REQUIRED_VARS CUPTI_LIBRARY CUPTI_INCLUDE_DIR)

if(CUPTI_FOUND)
  message(STATUS "Found CUPTI: ${CUPTI_LIBRARY}")
  message(STATUS "CUPTI include dir: ${CUPTI_INCLUDE_DIR}")

  # Create an imported target for CUPTI
  if(NOT TARGET CUPTI::cupti)
    add_library(CUPTI::cupti SHARED IMPORTED)
    set_target_properties(
      CUPTI::cupti PROPERTIES IMPORTED_LOCATION "${CUPTI_LIBRARY}" INTERFACE_INCLUDE_DIRECTORIES
                                                                   "${CUPTI_INCLUDE_DIR}"
    )
  endif()
else()
  message(FATAL_ERROR "CUPTI not found. Please ensure CUDA toolkit with CUPTI is installed.")
endif()

mark_as_advanced(CUPTI_INCLUDE_DIR CUPTI_LIBRARY)
