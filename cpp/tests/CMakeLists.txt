# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =================================================================================

# ##################################################################################################
# enable testing -----------------------------------------------------------------------------------
# ##################################################################################################
enable_testing()

# Create separate test case for each of this n_ranks
set(nranks_to_run 1 2 3 4 5 8)

# A helper function to create a test case to run each parallelism (defined by n_ranks_to_run).
# TEST_TARGET is the name of the test target (required). This will add len(n_ranks_to_run) test
# cases identified by <TEST_TARGET>_<n_ranks>. NOTE: When running ctest, the TEST_TARGET executable
# should be located in the ./gtests/ directory relative to the PWD.
function(rapidsmpf_mpirun_test_add)
  set(options) # no options
  set(one_value TEST_TARGET)
  set(multi_value) # no multi_value args
  cmake_parse_arguments(_MPIRUN_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(NOT DEFINED _MPIRUN_TEST_TEST_TARGET)
    message(FATAL_ERROR "rapidsmpf_mpirun_test_add called without a test prefix")
  endif()

  message(STATUS "Adding mpirun test: ${_MPIRUN_TEST_TEST_TARGET} nranks: ${nranks_to_run}")
  foreach(np IN ITEMS ${nranks_to_run})
    # add the test using the relative path of the target
    add_test(
      NAME "${_MPIRUN_TEST_TEST_TARGET}_${np}"
      COMMAND mpirun --map-by=node --bind-to=none -np=${np} "gtests/${_MPIRUN_TEST_TEST_TARGET}"
    )
  endforeach(np)
endfunction(rapidsmpf_mpirun_test_add)

# test targets are located in ${RAPIDSMPF_BINARY_DIR}/gtests, but ctest working directory would be
# ${RAPIDSMPF_BINARY_DIR}/tests. Therefore, a symlink (${RAPIDSMPF_BINARY_DIR}/tests/gtests ->
# ${RAPIDSMPF_BINARY_DIR}/gtests) is created to seamlessly run ctest from ${RAPIDSMPF_BINARY_DIR}.
file(CREATE_LINK "${RAPIDSMPF_BINARY_DIR}/gtests" "${CMAKE_CURRENT_BINARY_DIR}/gtests" SYMBOLIC)

add_library(test_sources INTERFACE)
target_sources(
  test_sources
  INTERFACE error_macros_tests.cpp
            test_allgather.cpp
            test_buffer_resource.cpp
            test_chunk.cpp
            test_communicator.cpp
            test_config.cpp
            test_cudf_utils.cpp
            test_cuda_stream.cpp
            test_cupti_monitor.cpp
            test_partition.cpp
            test_pausable_thread_loop.cpp
            test_pinned_resources.cpp
            test_progress_thread.cpp
            test_rmm_resource_adaptor.cpp
            test_shuffler.cpp
            test_shuffler_many_streams.cpp
            test_spill_manager.cpp
            test_statistics.cpp
            test_utils.cpp
)

# if streaming is enabled, add the streaming test sources
if(RAPIDSMPF_HAVE_STREAMING)
  target_sources(
    test_sources
    INTERFACE streaming/test_allgather.cpp
              streaming/test_error_handling.cpp
              streaming/test_leaf_node.cpp
              streaming/test_message.cpp
              streaming/test_partition.cpp
              streaming/test_shuffler.cpp
              streaming/test_table_chunk.cpp
  )
endif()

if(RAPIDSMPF_HAVE_MPI)
  add_executable(mpi_tests main/mpi.cpp)
  set_target_properties(
    mpi_tests
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RAPIDSMPF_BINARY_DIR}/gtests>"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               # For std:: support of __int128_t. Can be removed once using cuda::std
               CXX_EXTENSIONS ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               COMPONENT testing # custom test property, so that this could be searched later
               LINK_FLAGS "-Wl,--allow-shlib-undefined"
  )
  target_compile_options(
    mpi_tests PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
                      "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
  )
  target_link_libraries(
    mpi_tests
    PRIVATE rapidsmpf::rapidsmpf
            cudf::cudftestutil
            cudf::cudftestutil_impl
            GTest::gmock
            GTest::gtest
            $<TARGET_NAME_IF_EXISTS:MPI::MPI_C>
            $<TARGET_NAME_IF_EXISTS:conda_env>
            maybe_asan
            test_sources
            $<$<BOOL:${RAPIDSMPF_HAVE_NUMA}>:numa>
  )
  disable_sign_conversion_warning(mpi_tests)

  rapidsmpf_mpirun_test_add(TEST_TARGET mpi_tests)

  if(RAPIDSMPF_HAVE_UCXX)
    add_executable(ucxx_tests main/ucxx.cpp)
    set_target_properties(
      ucxx_tests
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RAPIDSMPF_BINARY_DIR}/gtests>"
                 CXX_STANDARD 20
                 CXX_STANDARD_REQUIRED ON
                 # For std:: support of __int128_t. Can be removed once using cuda::std
                 CXX_EXTENSIONS ON
                 CUDA_STANDARD 20
                 CUDA_STANDARD_REQUIRED ON
                 COMPONENT testing # custom test property, so that this could be searched later
                 LINK_FLAGS "-Wl,--allow-shlib-undefined"
    )
    target_compile_options(
      ucxx_tests PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
                         "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
    )
    target_link_libraries(
      ucxx_tests
      PRIVATE rapidsmpf::rapidsmpf
              cudf::cudftestutil
              cudf::cudftestutil_impl
              GTest::gmock
              GTest::gtest
              ucxx::ucxx
              $<TARGET_NAME_IF_EXISTS:MPI::MPI_C>
              $<TARGET_NAME_IF_EXISTS:conda_env>
              maybe_asan
              test_sources
              $<$<BOOL:${RAPIDSMPF_HAVE_NUMA}>:numa>
    )
    disable_sign_conversion_warning(ucxx_tests)

    rapidsmpf_mpirun_test_add(TEST_TARGET ucxx_tests)
  endif()
endif()

add_executable(single_tests main/single.cpp)

set_target_properties(
  single_tests
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RAPIDSMPF_BINARY_DIR}/gtests>"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             # For std:: support of __int128_t. Can be removed once using cuda::std
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             COMPONENT testing # custom test property, so that this could be searched later
)
target_compile_options(
  single_tests PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
                       "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
)
target_link_libraries(
  single_tests
  PRIVATE rapidsmpf::rapidsmpf
          cudf::cudftestutil
          cudf::cudftestutil_impl
          GTest::gmock
          GTest::gtest
          $<TARGET_NAME_IF_EXISTS:conda_env>
          $<$<BOOL:${RAPIDSMPF_HAVE_NUMA}>:numa>
          maybe_asan
          test_sources
)
disable_sign_conversion_warning(single_tests)
add_test(NAME single_tests COMMAND "gtests/single_tests")

# Create a list of targets that contains property KEY:VALUE.
function(rapidsmpf_targets_with_property)
  set(options)
  set(one_value KEY VALUE OUTPUT)
  set(multi_value COMMAND)
  cmake_parse_arguments(_RAPIDSMPF_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(NOT DEFINED _RAPIDSMPF_TEST_KEY)
    message(FATAL_ERROR "rapidsmpf_targets_with_property called without a KEY")
  endif()

  if(NOT DEFINED _RAPIDSMPF_TEST_VALUE)
    message(FATAL_ERROR "rapidsmpf_targets_with_property called without a VALUE")
  endif()

  if(NOT DEFINED _RAPIDSMPF_TEST_OUTPUT)
    message(FATAL_ERROR "rapidsmpf_targets_with_property called without a OUTPUT")
  endif()

  # get all targets
  get_property(
    all_targets
    DIRECTORY
    PROPERTY BUILDSYSTEM_TARGETS
  )

  set(targets_with_property "")
  foreach(target ${all_targets})
    get_target_property(prop_value ${target} ${_RAPIDSMPF_TEST_KEY})
    if(prop_value STREQUAL ${_RAPIDSMPF_TEST_VALUE})
      list(APPEND targets_with_property ${target})
    endif()
  endforeach()

  set(${_RAPIDSMPF_TEST_OUTPUT}
      ${targets_with_property}
      PARENT_SCOPE
  )
endfunction(rapidsmpf_targets_with_property)

# Search for all targets that has COMPONENT:testing property
rapidsmpf_targets_with_property(KEY COMPONENT VALUE testing OUTPUT testing_targets)

# Install testing targets to bin/tests/librapidsmpf/gtests
install(
  TARGETS ${testing_targets}
  DESTINATION bin/tests/librapidsmpf/gtests
  COMPONENT testing
  EXCLUDE_FROM_ALL
)

# Install the CTestTestfile.cmake to bin/tests/librapidsmpf
install(
  FILES "${RAPIDSMPF_BINARY_DIR}/tests/CTestTestfile.cmake"
  DESTINATION bin/tests/librapidsmpf
  COMPONENT testing
  EXCLUDE_FROM_ALL
)
