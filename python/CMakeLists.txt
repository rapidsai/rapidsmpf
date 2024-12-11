# =============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

include(../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(${rapids-cmake-dir}/cuda/set_runtime.cmake)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(pyrmp)

project(
  pyrmp
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX
)

find_package(RAPIDSMP "${RAPIDS_VERSION}" REQUIRED)

# ##################################################################################################
# * dependencies ----------------------------------------------------------------------------------
rapids_cpm_init()

include(rapids-cython-core)

rapids_cython_init()

add_subdirectory(pyrmp)
