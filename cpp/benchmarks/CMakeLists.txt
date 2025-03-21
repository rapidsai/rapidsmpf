# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =================================================================================

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

add_executable(bench_comm "bench_comm.cpp" ${UTILS_SOURCES})
set_target_properties(
  bench_comm
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RAPIDSMP_BINARY_DIR}/benchmarks>"
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             # For std:: support of __int128_t. Can be removed once using cuda::std
             CXX_EXTENSIONS ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  bench_comm PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMP_CXX_FLAGS}>"
                     "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMP_CUDA_FLAGS}>"
)
target_link_libraries(
  bench_comm PRIVATE rapidsmp::rapidsmp MPI::MPI_C ucxx::ucxx ucx::ucp
                     $<TARGET_NAME_IF_EXISTS:conda_env>
)
install(
  TARGETS bench_comm
  COMPONENT testing
  DESTINATION bin/benchmarks/librapidsmp
  EXCLUDE_FROM_ALL
)
