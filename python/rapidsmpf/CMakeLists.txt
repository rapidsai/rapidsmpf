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

rapids_cuda_init_architectures(rapidsmpf-python)

project(
  rapidsmpf-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# Add global Address Sanitizer flags only if explicitly requested
option(RAPIDSMPF_PYTHON_ASAN "Enable Address Sanitizer for Python build" OFF)
add_library(maybe_asan INTERFACE)
target_compile_options(
  maybe_asan INTERFACE "$<$<COMPILE_LANGUAGE:C>:$<$<BOOL:${RAPIDSMPF_ASAN}>:-fsanitize=address>>"
                       "$<$<COMPILE_LANGUAGE:CXX>:$<$<BOOL:${RAPIDSMPF_ASAN}>:-fsanitize=address>>"
)
target_link_options(maybe_asan INTERFACE "$<$<BOOL:${RAPIDSMPF_ASAN}>:-fsanitize=address>")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(rapidsmpf REQUIRED "${RAPIDS_VERSION}")

rapids_find_package(ucxx-python REQUIRED "${UCXX_VERSION}")
find_package(CUDAToolkit REQUIRED)

include(rapids-cython-core)
rapids_cython_init()

set(cython_lib_dir rapidsmpf)

add_subdirectory(rapidsmpf)
