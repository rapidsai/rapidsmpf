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

add_subdirectory(../../cpp librapidsmpf)

# Or it can't find `ucxx` but this seems wrong
set(ucxx-python_DIR /pyenv/versions/3.13.3/lib/python3.13/site-packages/ucxx/lib64/cmake/ucxx-python/)

# Or it can't find <ucxx/python/exception.h> but again seems wrong
include_directories("/pyenv/versions/3.13.3/lib/python3.13/site-packages/ucxx/include")

rapids_find_package(ucx REQUIRED)
rapids_find_package(ucxx REQUIRED "${UCXX_VERSION}")
rapids_find_package(ucxx-python REQUIRED "${UCXX_VERSION}")
find_package(CUDAToolkit REQUIRED)


include(rapids-cython-core)
rapids_cython_init()

set(cython_lib_dir rapidsmpf)

add_subdirectory(rapidsmpf)
