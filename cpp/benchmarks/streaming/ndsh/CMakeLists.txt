# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =================================================================================

if(NOT RAPIDSMPF_HAVE_MPI)
  message(FATAL_ERROR "Streaming NDSH benchmarks require MPI support")
endif()

if(NOT RAPIDSMPF_HAVE_STREAMING)
  message(FATAL_ERROR "Streaming NDSH benchmarks require streaming support")
endif()

add_library(
  rapidsmpfndsh bloom_filter.cpp bloom_filter_impl.cu concatenate.cpp groupby.cpp join.cpp
                parquet_writer.cpp sort.cpp utils.cpp
)

set_target_properties(
  rapidsmpfndsh
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(
  rapidsmpfndsh PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
                        "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
)
target_link_libraries(
  rapidsmpfndsh PRIVATE rapidsmpf::rapidsmpf $<TARGET_NAME_IF_EXISTS:MPI::MPI_C>
                        $<TARGET_NAME_IF_EXISTS:conda_env> maybe_asan
)

add_executable(q01 "q01.cpp")
set_target_properties(
  q01
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RAPIDSMPF_BINARY_DIR}/benchmarks/ndsh>"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  q01 PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
              "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
)
target_link_libraries(
  q01 PRIVATE rapidsmpfndsh rapidsmpf::rapidsmpf $<TARGET_NAME_IF_EXISTS:MPI::MPI_C>
              $<TARGET_NAME_IF_EXISTS:conda_env> maybe_asan
)
add_executable(q03 "q03.cpp")
set_target_properties(
  q03
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RAPIDSMPF_BINARY_DIR}/benchmarks/ndsh>"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  q03 PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
              "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
)
target_link_libraries(
  q03 PRIVATE rapidsmpfndsh rapidsmpf::rapidsmpf $<TARGET_NAME_IF_EXISTS:MPI::MPI_C>
              $<TARGET_NAME_IF_EXISTS:conda_env> maybe_asan
)
add_executable(q09 "q09.cpp")
set_target_properties(
  q09
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RAPIDSMPF_BINARY_DIR}/benchmarks/ndsh>"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  q09 PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
              "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
)
target_link_libraries(
  q09 PRIVATE rapidsmpfndsh rapidsmpf::rapidsmpf $<TARGET_NAME_IF_EXISTS:MPI::MPI_C>
              $<TARGET_NAME_IF_EXISTS:conda_env> maybe_asan
)

install(
  TARGETS rapidsmpfndsh
  COMPONENT benchmarking
  DESTINATION ${lib_dir}
  EXCLUDE_FROM_ALL
)
install(
  TARGETS q01
  COMPONENT benchmarking
  DESTINATION bin/benchmarks/librapidsmpf
  EXCLUDE_FROM_ALL
)
install(
  TARGETS q03
  COMPONENT benchmarking
  DESTINATION bin/benchmarks/librapidsmpf
  EXCLUDE_FROM_ALL
)
install(
  TARGETS q09
  COMPONENT benchmarking
  DESTINATION bin/benchmarks/librapidsmpf
  EXCLUDE_FROM_ALL
)
