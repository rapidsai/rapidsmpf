# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =================================================================================

add_executable(example_shuffle "example_shuffle.cpp" "../benchmarks/utils/random_data.cu")
set_target_properties(
  example_shuffle
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RAPIDSMPF_BINARY_DIR}/examples>"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             # For std:: support of __int128_t. Can be removed once using cuda::std
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  example_shuffle
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
          "$<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
)
target_link_libraries(
  example_shuffle PRIVATE rapidsmpf::rapidsmpf ucxx::ucxx ucx::ucp
                          $<TARGET_NAME_IF_EXISTS:MPI::MPI_C> $<TARGET_NAME_IF_EXISTS:conda_env>
)
target_link_options(example_shuffle PRIVATE -fsanitize=address)
install(
  TARGETS example_shuffle
  COMPONENT testing
  DESTINATION bin/examples/librapidsmpf
  EXCLUDE_FROM_ALL
)
