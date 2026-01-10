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

# cuco is not -Wsign-conversion safe
set_source_files_properties(bloom_filter_impl.cu PROPERTIES COMPILE_OPTIONS "-Wno-sign-conversion")
target_compile_options(
  rapidsmpfndsh PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
                        "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
)
target_link_libraries(
  rapidsmpfndsh PRIVATE rapidsmpf::rapidsmpf $<TARGET_NAME_IF_EXISTS:MPI::MPI_C>
                        $<TARGET_NAME_IF_EXISTS:conda_env> maybe_asan
)

set(RAPIDSMPFNDSH_QUERIES q01 q03 q09 q21)

foreach(query IN ITEMS ${RAPIDSMPFNDSH_QUERIES})
  add_executable(${query} "${query}.cpp")
  set_target_properties(
    ${query}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RAPIDSMPF_BINARY_DIR}/benchmarks/ndsh>"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
  )
  target_compile_options(
    ${query} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
                     "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
  )
  target_link_libraries(
    ${query} PRIVATE rapidsmpfndsh rapidsmpf::rapidsmpf $<TARGET_NAME_IF_EXISTS:MPI::MPI_C>
                     $<TARGET_NAME_IF_EXISTS:conda_env> maybe_asan
  )
endforeach()

install(
  TARGETS rapidsmpfndsh
  COMPONENT benchmarking
  DESTINATION ${lib_dir}
  EXCLUDE_FROM_ALL
)
install(
  TARGETS ${RAPIDSMPFNDSH_QUERIES}
  COMPONENT benchmarking
  DESTINATION bin/benchmarks/librapidsmpf
  EXCLUDE_FROM_ALL
)
