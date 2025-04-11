# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =================================================================================

add_executable(example_shuffle "example_shuffle.cpp" "../benchmarks/utils/random_data.cu")
set_target_properties(
  example_shuffle
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RAPIDSMP_BINARY_DIR}/examples>"
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             # For std:: support of __int128_t. Can be removed once using cuda::std
             CXX_EXTENSIONS ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  example_shuffle PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMP_CXX_FLAGS}>"
                          "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMP_CUDA_FLAGS}>"
)
target_link_libraries(
  example_shuffle PRIVATE rapidsmpf::rapidsmp MPI::MPI_C ucxx::ucxx ucx::ucp
                          $<TARGET_NAME_IF_EXISTS:conda_env>
)
install(
  TARGETS example_shuffle
  COMPONENT testing
  DESTINATION bin/examples/librapidsmp
  EXCLUDE_FROM_ALL
)
