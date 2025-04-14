# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =================================================================================

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cpm)
rapids_cpm_init()
include(rapids-cuda)
include(rapids-find)
rapids_cuda_init_architectures(rapidsmpf-python)

project(
  rapidsmpf-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

find_package(rapidsmp REQUIRED "${RAPIDS_VERSION}")

find_package(CUDAToolkit REQUIRED)

include(rapids-cython-core)
rapids_cython_init()

set(cython_lib_dir rapidsmpf)

add_subdirectory(rapidsmpf)
