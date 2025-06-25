# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =================================================================================

set(cython_modules config.pyx progress_thread.pyx rmm_resource_adaptor.pyx shuffler.pyx
                   statistics.pyx
)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_modules}"
  LINKED_LIBRARIES rapidsmpf::rapidsmpf
)

# Add Address Sanitizer to the generated targets
target_link_options(config PRIVATE "-fsanitize=address")
target_link_options(progress_thread PRIVATE "-fsanitize=address")
target_link_options(rmm_resource_adaptor PRIVATE "-fsanitize=address")
target_link_options(shuffler PRIVATE "-fsanitize=address")
target_link_options(statistics PRIVATE "-fsanitize=address")

add_subdirectory(_detail)
add_subdirectory(buffer)
add_subdirectory(communicator)
add_subdirectory(integrations/cudf)
add_subdirectory(utils)
