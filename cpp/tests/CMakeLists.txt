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

# Use an MPI-aware vendored version of the resource spec generation. Since we run ctest with
# `mpiexec -n NUM_PROCESSES ctest` we need to make the resource spec generation only produce output
# on a single rank.
set(rapidsmp_test_spec_file "${RAPIDSMP_BINARY_DIR}/resource_spec.json")
# This next bit does what rapids_test_init does to generate the CTEST_RESOURCE_SPEC_FILE variable.
# But we need to change the file being compiled.
add_executable(generate_resource_json "generate_resource_spec.cpp")
set_target_properties(
  generate_resource_json
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RAPIDSMP_BINARY_DIR}/gtests>"
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  generate_resource_json PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMP_CXX_FLAGS}>"
                                 "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMP_CUDA_FLAGS}>"
)
target_link_libraries(generate_resource_json PRIVATE CUDA::cudart MPI::MPI_C)

add_test(NAME generate_resource_spec COMMAND generate_resource_json "${rapidsmp_test_spec_file}")
set_tests_properties(
  generate_resource_spec PROPERTIES FIXTURES_SETUP resource_spec GENERATE_RESOURCE_SPEC_FILE
                                    "${rapidsmp_test_spec_file}"
)
add_library(rapids_test_install_testing INTERFACE)
set_property(
  TARGET rapids_test_install_testing
  APPEND
  PROPERTY "TESTS_TO_RUN" generate_resource_json
)
set_property(
  TARGET rapids_test_install_testing
  APPEND
  PROPERTY "TARGETS_TO_INSTALL" generate_resource_json
)

set(CTEST_RESOURCE_SPEC_FILE "${rapidsmp_test_spec_file}")
include(rapids-test)
rapids_test_init()

set(RAPIDSMP_GENERIC_TEST_SOURCES test_buffer_resource.cpp test_shuffler.cpp test_statistics.cpp)
add_executable(mpi_tests main/mpi.cpp "${RAPIDSMP_GENERIC_TEST_SOURCES}")
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
rapids_test_add(
  NAME mpi_tests
  COMMAND mpi_tests
  GPUS 1
  INSTALL_COMPONENT_SET testing
)

add_executable(ucxx_tests "main/ucxx.cpp" "${RAPIDSMP_GENERIC_TEST_SOURCES}")
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
rapids_test_add(
  NAME ucxx_tests
  COMMAND ucxx_tests
  GPUS 1
  INSTALL_COMPONENT_SET testing
)

# Install tests
rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/tests/librapidsmp)
