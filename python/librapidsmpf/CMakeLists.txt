# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =================================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cpm)
rapids_cpm_init()
include(rapids-cuda)
include(rapids-find)

rapids_cuda_init_architectures(librapidsmpf)

project(
  librapidsmpf-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# For now, disable CMake's automatic module scanning for C++ files. There is an sccache bug in the
# version RAPIDS uses in CI that causes it to handle the resulting -M* flags incorrectly with
# gcc>=14. We can remove this once we upgrade to a newer sccache version.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

find_package(rapidsmpf "${RAPIDS_VERSION}")

if(rapidsmpf_FOUND)
  return()
endif()

unset(rapidsmpf_FOUND)

set(CUDA_STATIC_RUNTIME ON)
add_subdirectory(../../cpp librapidsmpf-cpp)
find_package(CUDAToolkit REQUIRED)
