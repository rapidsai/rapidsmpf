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
include(${rapids-cmake-dir}/cython-core/find_prefix_paths.cmake)

rapids_cuda_init_architectures(rapidsmpf-python)

project(
  rapidsmpf-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# Add global AddressSanitizer flags only if explicitly requested
option(RAPIDSMPF_PYTHON_ASAN "Enable AddressSanitizer for Python build" OFF)
add_library(maybe_asan INTERFACE)
target_compile_options(
  maybe_asan
  INTERFACE "$<$<COMPILE_LANGUAGE:C,CXX>:$<$<BOOL:${RAPIDSMPF_PYTHON_ASAN}>:-fsanitize=address>>"
)
target_link_options(maybe_asan INTERFACE "$<$<BOOL:${RAPIDSMPF_PYTHON_ASAN}>:-fsanitize=address>")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For now, disable CMake's automatic module scanning for C++ files. There is an sccache bug in the
# version RAPIDS uses in CI that causes it to handle the resulting -M* flags incorrectly with
# gcc>=14. We can remove this once we upgrade to a newer sccache version.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# ##################################################################################################
# * find C++ libraries from RAPIDS wheels in in site-packages --------------------------------------
find_package(Python COMPONENTS Interpreter)
if(Python_FOUND)
  rapids_cython_find_prefix_paths("${Python_EXECUTABLE}" wheel_prefix_paths)
  foreach(wheel_prefix_path IN LISTS wheel_prefix_paths)
    list(APPEND CMAKE_PREFIX_PATH "${wheel_prefix_path}/lib64/cmake")
  endforeach()
endif()

find_package(rapidsmpf REQUIRED "${RAPIDS_VERSION}")

if(RAPIDSMPF_HAVE_UCXX)
  rapids_find_package(ucxx-python REQUIRED "${UCXX_VERSION}")
endif()
find_package(CUDAToolkit REQUIRED)

include(rapids-cython-core)
rapids_cython_init()

set(cython_lib_dir rapidsmpf)

add_subdirectory(rapidsmpf)
