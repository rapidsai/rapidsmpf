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

rapids_cuda_init_architectures(RAPIDSMP)

project(
  RAPIDSMP
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES C CXX CUDA
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

# Write the version header
rapids_cmake_write_version_file(include/rapidsmp/version_config.hpp)

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# ##################################################################################################
# * build options ---------------------------------------------------------------------------------

option(BUILD_TESTS "Configure CMake to build tests" ON)
option(BUILD_BENCHMARKS "Configure CMake to build benchmarks" ON)
option(BUILD_EXAMPLES "Configure CMake to build examples" ON)
option(BUILD_SHARED_LIBS "Build rapids-mp shared library" ON)
# cudart can be statically linked or dynamically linked. The python ecosystem wants dynamic linking
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)
option(RAPIDSMP_STATIC_LINTERS "Enable static linters (e.g. clang-tidy) during compilation" OFF)

set(RAPIDSMP_BUILD_TESTS ${BUILD_TESTS})
set(RAPIDSMP_BUILD_BENCHMARKS ${BUILD_BENCHMARKS})
set(RAPIDSMP_BUILD_EXAMPLES ${BUILD_EXAMPLES})

# ##################################################################################################
# * conda environment -----------------------------------------------------------------------------
rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

# ##################################################################################################
# * compiler options ------------------------------------------------------------------------------
rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET rapidsmp-exports
  INSTALL_EXPORT_SET rapidsmp-exports
)
include(../cmake/Modules/ConfigureCUDA.cmake) # set other CUDA compilation flags

# ##################################################################################################
# * dependencies ----------------------------------------------------------------------------------
rapids_cpm_init()

rapids_find_package(
  Threads REQUIRED
  BUILD_EXPORT_SET rapidsmp-exports
  INSTALL_EXPORT_SET rapidsmp-exports
)

include(../cmake/thirdparty/get_nvtx.cmake)

include(../cmake/thirdparty/get_rmm.cmake)
include(../cmake/thirdparty/get_cudf.cmake)
if(RAPIDSMP_BUILD_TESTS)
  include(../cmake/thirdparty/get_gtest.cmake)
  rapids_cpm_find(MPI REQUIRED)
endif()

# ##################################################################################################
# * library targets --------------------------------------------------------------------------------

file(GLOB SOURCES "src/*.cpp" "src/shuffler/*.cpp" "src/communicator/*.cpp")
add_library(rapidsmp ${SOURCES})

set_target_properties(
  rapidsmp
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             # set target compile options
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             # For std:: support of __int128_t. Can be removed once using cuda::std
             CXX_EXTENSIONS ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(
  rapidsmp PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMP_CXX_FLAGS}>"
                   "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMP_CUDA_FLAGS}>"
)

target_include_directories(
  rapidsmp
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  INTERFACE "$<INSTALL_INTERFACE:include>"
)

target_link_libraries(rapidsmp PUBLIC rmm::rmm cudf::cudf)

# Add Conda library, and include paths if specified
if(TARGET conda_env)
  target_link_libraries(rapidsmp PRIVATE conda_env)
endif()

rapids_cuda_set_runtime(rapidsmp USE_STATIC ${CUDA_STATIC_RUNTIME})

add_library(rapidsmp::rapidsmp ALIAS rapidsmp)

# ##################################################################################################
# * linter configuration ---------------------------------------------------------------------------
if(RAPIDSMP_STATIC_LINTERS)
  if(CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
  endif()

  # For simplicity, for now we assume that all linters can be installed into an environment where
  # any linter is being run. We could relax this requirement if desired.
  find_program(
    CLANG_TIDY_EXE
    NAMES "clang-tidy"
    DOC "Path to clang-tidy executable" REQUIRED
  )

  execute_process(
    COMMAND ${CLANG_TIDY_EXE} --version
    OUTPUT_VARIABLE CLANG_TIDY_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  string(REGEX MATCH "LLVM version ([0-9]+\\.[0-9]+)\\.[0-9]+" LLVM_VERSION_MATCH
               "${CLANG_TIDY_OUTPUT}"
  )
  # Discard the patch version and allow it to float. Empirically, results between patch versions are
  # mostly stable, and looking at available packages on some package managers sometimes patch
  # versions are skipped so we don't want to constrain to a patch version that the user can't
  # install.
  set(LLVM_VERSION "${CMAKE_MATCH_1}")
  set(expected_clang_tidy_version 19.1)
  if(NOT expected_clang_tidy_version VERSION_EQUAL LLVM_VERSION)
    message(
      FATAL_ERROR
        "clang-tidy version ${expected_clang_tidy_version} is required, but found ${LLVM_VERSION}"
    )
  endif()

  # clang will complain about unused link libraries on the compile line unless we specify
  # -Qunused-arguments.
  set_target_properties(
    rapidsmp
    PROPERTIES CXX_CLANG_TIDY
               "clang-tidy;-header-filter=^${CMAKE_SOURCE_DIR}/.*;--extra-arg=-Qunused-arguments"
  )
endif()

# ##################################################################################################
# * add tests -------------------------------------------------------------------------------------
if(RAPIDSMP_BUILD_TESTS)
  # include CTest module -- automatically calls enable_testing()
  include(CTest)

  # ctest cuda memcheck
  find_program(CUDA_SANITIZER compute-sanitizer)
  set(MEMORYCHECK_COMMAND ${CUDA_SANITIZER})
  set(MEMORYCHECK_TYPE CudaSanitizer)
  set(CUDA_SANITIZER_COMMAND_OPTIONS "--tool memcheck")

  # Always print verbose output when tests fail if run using `make test`.
  list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
  add_subdirectory(tests)
endif()

# ##################################################################################################
# * add benchmarks
# -------------------------------------------------------------------------------------
if(RAPIDSMP_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# ##################################################################################################
# * add examples
# -------------------------------------------------------------------------------------
if(RAPIDSMP_BUILD_BENCHMARKS)
  add_subdirectory(examples)
endif()

# ##################################################################################################
# * install targets -------------------------------------------------------------------------------
rapids_cmake_install_lib_dir(lib_dir)
include(CPack)
include(GNUInstallDirs)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME rapidsmp)

install(
  TARGETS rapidsmp
  DESTINATION ${lib_dir}
  EXPORT rapidsmp-exports
)
install(DIRECTORY include/rapidsmp/ DESTINATION include/rapidsmp)

set(doc_string
    [=[
RAPIDS-MP.
    ]=]
)

rapids_export(
  INSTALL rapidsmp
  EXPORT_SET rapidsmp-exports
  GLOBAL_TARGETS rapidsmp
  NAMESPACE rapidsmp::
  DOCUMENTATION doc_string
)

# ##################################################################################################
# * build export -------------------------------------------------------------------------------

rapids_export(
  BUILD rapidsmp
  EXPORT_SET rapidsmp-exports
  GLOBAL_TARGETS rapidsmp
  NAMESPACE rapidsmp::
  DOCUMENTATION doc_string
)
