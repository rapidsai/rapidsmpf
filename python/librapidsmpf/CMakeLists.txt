# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =================================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cpm)
rapids_cpm_init()
include(rapids-cuda)
include(rapids-find)

rapids_cuda_init_architectures(librapidsmpf)

project(
  librapidsmpf
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

find_package(librapidsmpf "${RAPIDS_VERSION}")

if(librapidsmpf_FOUND)
  return()
endif()

unset(librapidsmpf_FOUND)

set(CUDA_STATIC_RUNTIME ON)
add_subdirectory(../../cpp librapidsmpf)
find_package(CUDAToolkit REQUIRED)
