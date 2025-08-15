# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =================================================================================

add_executable(bench_streaming_shuffle "bench_streaming_shuffle.cpp" ${UTILS_SOURCES})
set_target_properties(
  bench_streaming_shuffle
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RAPIDSMPF_BINARY_DIR}/benchmarks>"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             # For std:: support of __int128_t. Can be removed once using cuda::std
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  bench_streaming_shuffle PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
                                  "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
)
target_link_libraries(
  bench_streaming_shuffle
  PRIVATE rapidsmpf::rapidsmpf ucxx::ucxx ucx::ucp $<TARGET_NAME_IF_EXISTS:MPI::MPI_C>
          $<TARGET_NAME_IF_EXISTS:conda_env> maybe_asan
)
install(
  TARGETS bench_streaming_shuffle
  COMPONENT benchmarking
  DESTINATION bin/benchmarks/librapidsmpf
  EXCLUDE_FROM_ALL
)
