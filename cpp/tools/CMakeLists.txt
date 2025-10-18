# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =================================================================================

# Topology Discovery Tool
add_executable(topology_discovery topology_discovery.cpp)

set_target_properties(topology_discovery PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

target_compile_options(
  topology_discovery PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
)

# Link against NVML library
target_link_libraries(topology_discovery PRIVATE CUDA::nvml $<TARGET_NAME_IF_EXISTS:conda_env>)

# Install the tool
install(TARGETS topology_discovery DESTINATION bin)
