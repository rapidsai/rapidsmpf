# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =================================================================================

add_executable(
  rrun
  "rrun.cpp"
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/system_info.cpp>"
  "$<$<BOOL:${RAPIDSMPF_HAVE_SLURM}>:$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/bootstrap/slurm_backend.cpp>>"
)
set_target_properties(
  rrun
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${RAPIDSMPF_BINARY_DIR}/tools"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
)
target_include_directories(rrun PRIVATE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")
target_compile_options(rrun PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>")
target_compile_definitions(
  rrun PRIVATE $<$<BOOL:${RAPIDSMPF_HAVE_NUMA}>:RAPIDSMPF_HAVE_NUMA>
               $<$<BOOL:${RAPIDSMPF_HAVE_SLURM}>:RAPIDSMPF_HAVE_SLURM>
)
target_link_libraries(
  rrun
  PRIVATE Threads::Threads
          cuCascade::cucascade
          $<$<TARGET_EXISTS:kvikio::kvikio>:kvikio::kvikio>
          $<TARGET_NAME_IF_EXISTS:conda_env>
          maybe_asan
          $<$<BOOL:${RAPIDSMPF_HAVE_NUMA}>:numa>
          $<TARGET_NAME_IF_EXISTS:PMIx::PMIx>
          ${CMAKE_DL_LIBS}
)
install(
  TARGETS rrun
  COMPONENT tools
  DESTINATION bin
  EXCLUDE_FROM_ALL
)

add_executable(
  topology_discovery "topology_discovery.cpp"
                     "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/system_info.cpp>"
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
target_link_libraries(
  topology_discovery
  PRIVATE cuCascade::cucascade
          $<$<TARGET_EXISTS:kvikio::kvikio>:kvikio::kvikio>
          $<TARGET_NAME_IF_EXISTS:conda_env>
          $<$<BOOL:${RAPIDSMPF_HAVE_NUMA}>:numa>
          maybe_asan
          ${CMAKE_DL_LIBS}
)
install(
  TARGETS topology_discovery
  COMPONENT tools
  DESTINATION bin
  EXCLUDE_FROM_ALL
)

add_executable(
  check_resource_binding
  "check_resource_binding.cpp" "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/bootstrap/utils.cpp>"
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/system_info.cpp>"
)
set_target_properties(
  check_resource_binding
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${RAPIDSMPF_BINARY_DIR}/tools"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
)
target_include_directories(
  check_resource_binding PRIVATE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)
target_compile_options(
  check_resource_binding PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
)
target_compile_definitions(
  check_resource_binding PRIVATE $<$<BOOL:${RAPIDSMPF_HAVE_NUMA}>:RAPIDSMPF_HAVE_NUMA>
)
target_link_libraries(
  check_resource_binding
  PRIVATE cuCascade::cucascade
          $<$<TARGET_EXISTS:kvikio::kvikio>:kvikio::kvikio>
          $<TARGET_NAME_IF_EXISTS:conda_env>
          maybe_asan
          $<$<BOOL:${RAPIDSMPF_HAVE_NUMA}>:numa>
          ${CMAKE_DL_LIBS}
)
install(
  TARGETS check_resource_binding
  COMPONENT tools
  DESTINATION bin
  EXCLUDE_FROM_ALL
)
