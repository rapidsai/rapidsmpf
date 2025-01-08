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

file(GLOB UTILS_SOURCES "utils/*.cpp" "utils/*.cu")
add_executable(bench_shuffle "bench_shuffle.cpp" ${UTILS_SOURCES})
set_target_properties(
  bench_shuffle
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RAPIDSMP_BINARY_DIR}/benchmarks>"
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             # For std:: support of __int128_t. Can be removed once using cuda::std
             CXX_EXTENSIONS ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  bench_shuffle PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMP_CXX_FLAGS}>"
                        "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMP_CUDA_FLAGS}>"
)
target_link_libraries(
  bench_shuffle PRIVATE rapidsmp::rapidsmp MPI::MPI_C ucxx::ucxx ucx::ucp
                        $<TARGET_NAME_IF_EXISTS:conda_env>
)
install(
  TARGETS bench_shuffle
  COMPONENT testing
  DESTINATION bin/benchmarks/librapidsmp
  EXCLUDE_FROM_ALL
)

add_executable(bench_shuffle_ucxx "bench_shuffle_ucxx.cpp" ${UTILS_SOURCES})
set_target_properties(
  bench_shuffle_ucxx
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RAPIDSMP_BINARY_DIR}/benchmarks>"
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             # For std:: support of __int128_t. Can be removed once using cuda::std
             CXX_EXTENSIONS ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  bench_shuffle_ucxx PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMP_CXX_FLAGS}>"
                             "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMP_CUDA_FLAGS}>"
)
target_link_libraries(
  bench_shuffle_ucxx PRIVATE rapidsmp::rapidsmp MPI::MPI_C ucxx::ucxx ucx::ucp
                             $<TARGET_NAME_IF_EXISTS:conda_env>
)
install(
  TARGETS bench_shuffle_ucxx
  COMPONENT testing
  DESTINATION bin/benchmarks/librapidsmp
  EXCLUDE_FROM_ALL
)
