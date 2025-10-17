# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =================================================================================

# rrun launcher tool
add_executable(rrun "rrun.cpp")
set_target_properties(
  rrun
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RAPIDSMPF_BINARY_DIR}/tools>"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
)
target_compile_options(rrun PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>")
target_link_libraries(rrun PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env> maybe_asan)
install(
  TARGETS rrun
  COMPONENT tools
  DESTINATION bin
  EXCLUDE_FROM_ALL
)
