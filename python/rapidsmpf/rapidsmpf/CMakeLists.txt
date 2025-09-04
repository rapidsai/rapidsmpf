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
  LINKED_LIBRARIES rapidsmpf::rapidsmpf maybe_asan
)

add_subdirectory(_detail)
add_subdirectory(allgather)
add_subdirectory(buffer)
add_subdirectory(communicator)
add_subdirectory(integrations/cudf)
add_subdirectory(utils)

if(RAPIDSMPF_HAVE_STREAMING)
  add_subdirectory(streaming)
endif()
