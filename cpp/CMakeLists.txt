# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =================================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(${rapids-cmake-dir}/cuda/set_runtime.cmake)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(RAPIDSMPF)

project(
  RAPIDSMPF
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES C CXX CUDA
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

# For now, disable CMake's automatic module scanning for C++ files. There is an sccache bug in the
# version RAPIDS uses in CI that causes it to handle the resulting -M* flags incorrectly with
# gcc>=14. We can remove this once we upgrade to a newer sccache version.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Write the version header
rapids_cmake_write_version_file(include/rapidsmpf/version_config.hpp)

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# ##################################################################################################
# * build options ---------------------------------------------------------------------------------

option(BUILD_MPI_SUPPORT "Build RapidsMPF with MPI support" ON)
option(BUILD_UCXX_SUPPORT "Build RapidsMPF with UCXX support" ON)
option(BUILD_STREAMING_SUPPORT "Build RapidsMPF with streaming support" ON)
option(BUILD_CUPTI_SUPPORT "Build RapidsMPF with CUPTI support" OFF)
option(BUILD_NUMA_SUPPORT "Build RapidsMPF with NUMA support" ON)
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(BUILD_BENCHMARKS "Configure CMake to build benchmarks" ON)
option(BUILD_EXAMPLES "Configure CMake to build examples" ON)
option(BUILD_SHARED_LIBS "Build RapidsMPF shared library" ON)
# cudart can be statically linked or dynamically linked. The python ecosystem wants dynamic linking
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)
option(RAPIDSMPF_CLANG_TIDY "Enable clang-tidy during compilation" OFF)
option(RAPIDSMPF_ASAN "Enable AddressSanitizer" OFF)

message(STATUS "librapidsmpf build options:")
message(STATUS "  BUILD_MPI_SUPPORT       : ${BUILD_MPI_SUPPORT}")
message(STATUS "  BUILD_UCXX_SUPPORT      : ${BUILD_UCXX_SUPPORT}")
message(STATUS "  BUILD_STREAMING_SUPPORT : ${BUILD_STREAMING_SUPPORT}")
message(STATUS "  BUILD_CUPTI_SUPPORT     : ${BUILD_CUPTI_SUPPORT}")
message(STATUS "  BUILD_NUMA_SUPPORT      : ${BUILD_NUMA_SUPPORT}")
message(STATUS "  BUILD_TESTS             : ${BUILD_TESTS}")
message(STATUS "  BUILD_BENCHMARKS        : ${BUILD_BENCHMARKS}")
message(STATUS "  BUILD_EXAMPLES          : ${BUILD_EXAMPLES}")
message(STATUS "  BUILD_SHARED_LIBS       : ${BUILD_SHARED_LIBS}")
message(STATUS "  CUDA_STATIC_RUNTIME     : ${CUDA_STATIC_RUNTIME}")
message(STATUS "  RAPIDSMPF_CLANG_TIDY    : ${RAPIDSMPF_CLANG_TIDY}")
message(STATUS "  RAPIDSMPF_ASAN          : ${RAPIDSMPF_ASAN}")

# Copy options to our prefix to prevent upstream projects from modifying them.
set(RAPIDSMPF_HAVE_MPI ${BUILD_MPI_SUPPORT})
set(RAPIDSMPF_HAVE_UCXX ${BUILD_UCXX_SUPPORT})
set(RAPIDSMPF_HAVE_STREAMING ${BUILD_STREAMING_SUPPORT})
set(RAPIDSMPF_HAVE_CUPTI ${BUILD_CUPTI_SUPPORT})
set(RAPIDSMPF_HAVE_NUMA ${BUILD_NUMA_SUPPORT})
set(RAPIDSMPF_BUILD_TESTS ${BUILD_TESTS})
set(RAPIDSMPF_BUILD_BENCHMARKS ${BUILD_BENCHMARKS})
set(RAPIDSMPF_BUILD_EXAMPLES ${BUILD_EXAMPLES})

if(NOT (RAPIDSMPF_HAVE_MPI AND RAPIDSMPF_HAVE_UCXX))
  if(RAPIDSMPF_BUILD_BENCHMARKS)
    message(
      FATAL_ERROR
        "BUILD_BENCHMARKS cannot be ON when BUILD_MPI_SUPPORT or BUILD_UCXX_SUPPORT is OFF"
    )
  endif()
  if(RAPIDSMPF_BUILD_EXAMPLES)
    message(
      FATAL_ERROR "BUILD_EXAMPLES cannot be ON when BUILD_MPI_SUPPORT or BUILD_UCXX_SUPPORT is OFF"
    )
  endif()
endif()

# ##################################################################################################
# * conda environment -----------------------------------------------------------------------------
rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

# ##################################################################################################
# * compiler options ------------------------------------------------------------------------------
rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET rapidsmpf-exports
  INSTALL_EXPORT_SET rapidsmpf-exports
)
include(../cmake/Modules/ConfigureCUDA.cmake) # set other CUDA compilation flags

# ##################################################################################################
# * dependencies ----------------------------------------------------------------------------------
rapids_cpm_init()

rapids_find_package(
  Threads REQUIRED
  BUILD_EXPORT_SET rapidsmpf-exports
  INSTALL_EXPORT_SET rapidsmpf-exports
)

# NOTE: get_cccl_cudax.cmake needs to be included before rmm and cudf, because both require CCCL
# (but without CUDAX). All of these will call rapids_cpm_cccl. So, if get_cccl_cudax is included
# after, the CCCL::cudax target will not be available.
include(../cmake/thirdparty/get_cccl_cudax.cmake)
include(../cmake/thirdparty/get_nvtx.cmake)
include(../cmake/thirdparty/get_rmm.cmake)
include(../cmake/thirdparty/get_cudf.cmake)
if(RAPIDSMPF_HAVE_UCXX)
  rapids_find_package(
    ucxx REQUIRED
    BUILD_EXPORT_SET rapidsmpf-exports
    INSTALL_EXPORT_SET rapidsmpf-exports
  )
endif()
if(RAPIDSMPF_HAVE_MPI)
  rapids_find_package(
    MPI REQUIRED
    BUILD_EXPORT_SET rapidsmpf-exports
    INSTALL_EXPORT_SET rapidsmpf-exports
  )
endif()
if(RAPIDSMPF_BUILD_TESTS)
  include(../cmake/thirdparty/get_gtest.cmake)
endif()
if(RAPIDSMPF_HAVE_STREAMING)
  include(../cmake/thirdparty/get_libcoro.cmake)
endif()

# ##################################################################################################
# * library targets --------------------------------------------------------------------------------

add_library(maybe_asan INTERFACE)
target_compile_options(
  maybe_asan INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:$<$<BOOL:${RAPIDSMPF_ASAN}>:-fsanitize=address>>"
)
target_link_options(maybe_asan INTERFACE "$<$<BOOL:${RAPIDSMPF_ASAN}>:-fsanitize=address>")

add_library(
  rapidsmpf
  src/allgather/allgather.cpp
  src/buffer/buffer.cpp
  src/buffer/pinned_memory_resource.cu
  src/buffer/resource.cpp
  src/buffer/spill_manager.cpp
  src/communicator/communicator.cpp
  src/communicator/single.cpp
  src/config.cpp
  src/cuda_event.cpp
  src/integrations/cudf/partition.cpp
  src/integrations/cudf/utils.cpp
  src/pausable_thread_loop.cpp
  src/progress_thread.cpp
  src/rmm_resource_adaptor.cpp
  src/shuffler/chunk.cpp
  src/shuffler/finish_counter.cpp
  src/shuffler/postbox.cpp
  src/shuffler/shuffler.cpp
  src/statistics.cpp
  src/utils.cpp
)
if(RAPIDSMPF_HAVE_STREAMING)
  target_sources(
    rapidsmpf
    PRIVATE src/streaming/coll/allgather.cpp
            src/streaming/coll/shuffler.cpp
            src/streaming/core/context.cpp
            src/streaming/core/leaf_node.cpp
            src/streaming/core/node.cpp
            src/streaming/cudf/partition.cpp
            src/streaming/cudf/table_chunk.cpp
  )
endif()
if(RAPIDSMPF_HAVE_UCXX)
  target_sources(rapidsmpf PRIVATE src/communicator/ucxx.cpp)
endif()
if(RAPIDSMPF_HAVE_MPI)
  target_sources(rapidsmpf PRIVATE src/communicator/mpi.cpp)
endif()
if(RAPIDSMPF_HAVE_MPI AND RAPIDSMPF_HAVE_UCXX)
  target_sources(rapidsmpf PRIVATE src/communicator/ucxx_utils.cpp)
endif()
if(RAPIDSMPF_HAVE_CUPTI)
  target_sources(rapidsmpf PRIVATE src/cupti.cpp)
endif()

set_target_properties(
  rapidsmpf
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             # set target compile options
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             # For std:: support of __int128_t. Can be removed once using cuda::std
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(
  rapidsmpf PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
                    "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
)

target_include_directories(
  rapidsmpf
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  INTERFACE "$<INSTALL_INTERFACE:include>"
)

target_link_libraries(
  rapidsmpf
  PUBLIC rmm::rmm cudf::cudf CCCL::CCCL $<TARGET_NAME_IF_EXISTS:ucxx::ucxx>
         $<TARGET_NAME_IF_EXISTS:libcoro>
  PRIVATE $<$<BOOL:${RAPIDSMPF_HAVE_NUMA}>:numa>
          $<TARGET_NAME_IF_EXISTS:MPI::MPI_C>
          $<$<BOOL:${RAPIDSMPF_HAVE_CUPTI}>:CUDA::cupti>
          $<TARGET_NAME_IF_EXISTS:conda_env>
          maybe_asan
          $<TARGET_NAME_IF_EXISTS:CCCL::cudax>
)

target_compile_definitions(
  rapidsmpf
  PUBLIC $<$<BOOL:${RAPIDSMPF_HAVE_MPI}>:RAPIDSMPF_HAVE_MPI>
         $<$<BOOL:${RAPIDSMPF_HAVE_UCXX}>:RAPIDSMPF_HAVE_UCXX>
         $<$<BOOL:${RAPIDSMPF_HAVE_STREAMING}>:RAPIDSMPF_HAVE_STREAMING>
         $<$<BOOL:${RAPIDSMPF_HAVE_CUPTI}>:RAPIDSMPF_HAVE_CUPTI>
         $<$<BOOL:${RAPIDSMPF_HAVE_NUMA}>:RAPIDSMPF_HAVE_NUMA>
)

rapids_cuda_set_runtime(rapidsmpf USE_STATIC ${CUDA_STATIC_RUNTIME})

add_library(rapidsmpf::rapidsmpf ALIAS rapidsmpf)

# ##################################################################################################
# * linter configuration ---------------------------------------------------------------------------
if(RAPIDSMPF_CLANG_TIDY)
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
  set(expected_clang_tidy_version 20.1)
  if(NOT expected_clang_tidy_version VERSION_EQUAL LLVM_VERSION)
    message(
      FATAL_ERROR
        "clang-tidy version ${expected_clang_tidy_version} is required, but found ${LLVM_VERSION}"
    )
  endif()

  # Restrict clang-tidy to only run on our own source/include dirs.
  set(CLANG_TIDY_HEADER_FILTER
      "^-?(${CMAKE_SOURCE_DIR}/src|${CMAKE_SOURCE_DIR}/include|${CMAKE_SOURCE_DIR}/benchmarks|${CMAKE_SOURCE_DIR}/tests)/.*"
  )

  # adding `-Qunused-arguments`, otherwise clang complains about unused link libraries.
  set_target_properties(
    rapidsmpf
    PROPERTIES
      CXX_CLANG_TIDY
      "${CLANG_TIDY_EXE};--header-filter=${CLANG_TIDY_HEADER_FILTER};--extra-arg=-Qunused-arguments"
  )
endif()

# ##################################################################################################
# * add tests -------------------------------------------------------------------------------------
if(RAPIDSMPF_BUILD_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()

# ##################################################################################################
# * add benchmarks
# -------------------------------------------------------------------------------------
if(RAPIDSMPF_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# ##################################################################################################
# * add examples
# -------------------------------------------------------------------------------------
if(RAPIDSMPF_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# ##################################################################################################
# * install targets -------------------------------------------------------------------------------
rapids_cmake_install_lib_dir(lib_dir)
include(CPack)
include(GNUInstallDirs)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME rapidsmpf)

install(
  TARGETS rapidsmpf
  DESTINATION ${lib_dir}
  EXPORT rapidsmpf-exports
)
install(DIRECTORY include/rapidsmpf/ DESTINATION include/rapidsmpf)

set(doc_string
    [=[
RAPIDSMPF.
    ]=]
)

# Export `RAPIDSMPF_HAVE_*` to let downstream projects know what is supported.
string(
  JOIN
  "\n"
  final_code_string
  "set(RAPIDSMPF_HAVE_MPI [=[${RAPIDSMPF_HAVE_MPI}]=])"
  "set(RAPIDSMPF_HAVE_UCXX [=[${RAPIDSMPF_HAVE_UCXX}]=])"
  "set(RAPIDSMPF_HAVE_STREAMING [=[${RAPIDSMPF_HAVE_STREAMING}]=])"
  "set(RAPIDSMPF_HAVE_CUPTI [=[${RAPIDSMPF_HAVE_CUPTI}]=])"
)

rapids_export(
  INSTALL rapidsmpf
  EXPORT_SET rapidsmpf-exports
  GLOBAL_TARGETS rapidsmpf
  NAMESPACE rapidsmpf::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK final_code_string
)

# ##################################################################################################
# * build export -------------------------------------------------------------------------------

rapids_export(
  BUILD rapidsmpf
  EXPORT_SET rapidsmpf-exports
  GLOBAL_TARGETS rapidsmpf
  NAMESPACE rapidsmpf::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK final_code_string
)
