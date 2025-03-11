# =============================================================================
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

# ##################################################################################################
# enable testing -----------------------------------------------------------------------------------
# ##################################################################################################
enable_testing()

# Create separate test case for each of this n_ranks
set(nranks_to_run 1 2 3 4 5 8)

# A helper function to create a test case to run each parallelism (defined by n_ranks_to_run).
# TEST_TARGET is the name of the test target (required). This will add len(n_ranks_to_run) test
# cases identified by <TEST_TARGET>_<n_ranks>
function(rapidsmp_mpirun_test_add)
  set(options) # no options
  set(one_value TEST_TARGET)
  set(multi_value) # no multi_value args
  cmake_parse_arguments(_MPIRUN_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(NOT DEFINED _MPIRUN_TEST_TEST_TARGET)
    message(FATAL_ERROR "rapidsmp_mpirun_test_add called without a test prefix")
  endif()

  message(STATUS "Adding mpirun test: ${_MPIRUN_TEST_TEST_TARGET} nranks: ${nranks_to_run}")
  foreach(np IN ITEMS ${nranks_to_run})
    add_test(
      NAME "${_MPIRUN_TEST_TEST_TARGET}_${np}"
      COMMAND mpirun --map-by=node --bind-to=none -np=${np}
              "$<BUILD_INTERFACE:${RAPIDSMP_BINARY_DIR}/gtests>/${_MPIRUN_TEST_TEST_TARGET}"
    )
  endforeach(np)
endfunction(rapidsmp_mpirun_test_add)

file(GLOB SOURCES "main/mpi.cpp" "*.cpp")
add_executable(mpi_tests ${SOURCES})
set_target_properties(
  mpi_tests
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RAPIDSMP_BINARY_DIR}/gtests>"
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             # For std:: support of __int128_t. Can be removed once using cuda::std
             CXX_EXTENSIONS ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  mpi_tests PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMP_CXX_FLAGS}>"
                    "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMP_CUDA_FLAGS}>"
)
target_link_libraries(
  mpi_tests PRIVATE rapidsmp::rapidsmp cudf::cudftestutil cudf::cudftestutil_impl GTest::gmock
                    GTest::gtest MPI::MPI_C ucxx::ucxx ucx::ucp $<TARGET_NAME_IF_EXISTS:conda_env>
)

rapidsmp_mpirun_test_add(TEST_TARGET mpi_tests)

file(GLOB SOURCES "main/ucxx.cpp" "*.cpp")
add_executable(ucxx_tests ${SOURCES})
set_target_properties(
  ucxx_tests
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RAPIDSMP_BINARY_DIR}/gtests>"
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             # For std:: support of __int128_t. Can be removed once using cuda::std
             CXX_EXTENSIONS ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  ucxx_tests PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMP_CXX_FLAGS}>"
                     "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMP_CUDA_FLAGS}>"
)
target_link_libraries(
  ucxx_tests PRIVATE rapidsmp::rapidsmp cudf::cudftestutil cudf::cudftestutil_impl GTest::gmock
                     GTest::gtest MPI::MPI_C ucxx::ucxx ucx::ucp $<TARGET_NAME_IF_EXISTS:conda_env>
)

rapidsmp_mpirun_test_add(TEST_TARGET ucxx_tests)

# Install the CTestTestfile.cmake to bin/tests/librapidsmp directory
install(
  FILES "${RAPIDSMP_BINARY_DIR}/tests/CTestTestfile.cmake"
  DESTINATION bin/tests/librapidsmp
  COMPONENT testing
)
