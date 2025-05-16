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

rapids_cuda_init_architectures(librapidsmpf)

#option(BUILD_IN_CONDA "Build RapidsMPF library using conda" ON)

project(
  librapidsmpf
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

set(CUDA_STATIC_RUNTIME ON)
add_subdirectory(../../cpp librapidsmpf)
find_package(CUDAToolkit REQUIRED)


# if(NOT BUILD_IN_CONDA)
#   set(RELOCATE_LIBRAPIDSMPF ON)
#   set(CUDA_STATIC_RUNTIME ON)
#   add_subdirectory(../../cpp librapidsmpf)

#   # TODO: don't use env-vars for this
#   include_directories($ENV{SITE_PACKAGES}/ucxx/include)
#   set(ucxx-python_DIR $ENV{SITE_PACKAGES}/ucxx/lib64/cmake/ucxx-python/)

#   rapids_find_package(ucx REQUIRED)
#   rapids_find_package(ucxx REQUIRED "${UCXX_VERSION}")
# else()
#   find_package(rapidsmpf REQUIRED "${RAPIDS_VERSION}")
# endif()

# rapids_find_package(ucxx-python REQUIRED "${UCXX_VERSION}")
# find_package(CUDAToolkit REQUIRED)

# include(rapids-cython-core)
# rapids_cython_init()

# set(cython_lib_dir rapidsmpf)

# add_subdirectory(rapidsmpf)
