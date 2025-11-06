# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =================================================================================

add_executable(rrun "rrun.cpp")
set_target_properties(
  rrun
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${RAPIDSMPF_BINARY_DIR}/tools"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
)
target_include_directories(rrun PRIVATE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")
target_compile_options(rrun PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>")
target_link_libraries(rrun PRIVATE Threads::Threads $<TARGET_NAME_IF_EXISTS:conda_env> maybe_asan)
install(
  TARGETS rrun
  COMPONENT tools
  DESTINATION bin
  EXCLUDE_FROM_ALL
)

add_executable(
  topology_discovery "topology_discovery.cpp"
                     "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/topology_discovery.cpp>"
)
set_target_properties(
  topology_discovery
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${RAPIDSMPF_BINARY_DIR}/tools"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
)
target_include_directories(
  topology_discovery PRIVATE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)
target_compile_options(
  topology_discovery PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
)
target_link_libraries(topology_discovery PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env> maybe_asan)
install(
  TARGETS topology_discovery
  COMPONENT tools
  DESTINATION bin
  EXCLUDE_FROM_ALL
)
