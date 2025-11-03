# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =================================================================================

# Topology Discovery CLI Tool
add_executable(topology_discovery topology_discovery.cpp)

set_target_properties(topology_discovery PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

target_compile_options(
  topology_discovery PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
)

# Link against rapidsmpf library (which includes topology discovery)
target_link_libraries(
  topology_discovery PRIVATE rapidsmpf::rapidsmpf $<TARGET_NAME_IF_EXISTS:conda_env>
)

# Install the tool
install(TARGETS topology_discovery DESTINATION bin)
