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

include(rapids-test)
rapids_test_init()

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
rapids_test_add(
  NAME mpi_tests
  COMMAND mpi_tests
  GPUS 1
  INSTALL_COMPONENT_SET testing
)

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
rapids_test_add(
  NAME ucxx_tests
  COMMAND ucxx_tests
  GPUS 1
  INSTALL_COMPONENT_SET testing
)

# Install tests
rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/tests/librapidsmp)
