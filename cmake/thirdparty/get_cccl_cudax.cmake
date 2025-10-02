# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =================================================================================

# Find or fetch CCCL (CUDA C++ Core Libraries)
function(find_and_configure_cccl_cudax)
  message(STATUS "Finding CCCL_CUDAX")

  enable_language(CUDA)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
  endif()

  set(CCCL_VERSION 3.1.0)
  set(CCCL_TAG v3.1.0-rc8)

  include("${rapids-cmake-dir}/cpm/find.cmake")
  rapids_cpm_find(
    CCCL_CUDAX ${CCCL_VERSION} COMPONENTS cudax
    GLOBAL_TARGETS cudax
    BUILD_EXPORT_SET rapidsmpf-exports
    CPM_ARGS
    GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
    GIT_TAG ${CCCL_TAG}
    GIT_SHALLOW TRUE
    OPTIONS "CCCL_ENABLE_UNSTABLE ON" "CCCL_ENABLE_CUDAX OFF" # We dont want to build the CUDAX
                                                              # components
  )

  if(CCCL_CUDAX_ADDED)
    message(STATUS "Found CCCL_CUDAX: ${CCCL_CUDAX_SOURCE_DIR}")
  else()
    message(ERROR "Unable to add CCCL_CUDAX")
  endif()

endfunction()

find_and_configure_cccl_cudax()
