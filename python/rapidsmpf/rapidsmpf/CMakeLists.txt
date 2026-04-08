# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =================================================================================

set(cython_modules config.pyx cuda_stream.pyx error.pyx progress_thread.pyx
                   rmm_resource_adaptor.pyx shuffler.pyx statistics.pyx
)

# Add cupti module conditionally if CUPTI support is enabled
if(RAPIDSMPF_HAVE_CUPTI)
  list(APPEND cython_modules cupti.pyx)
endif()

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_modules}"
  LINKED_LIBRARIES rapidsmpf::rapidsmpf maybe_asan
)

add_subdirectory(_detail)
add_subdirectory(bootstrap)
add_subdirectory(coll)
add_subdirectory(memory)
add_subdirectory(communicator)
add_subdirectory(integrations/cudf)
add_subdirectory(utils)

if(RAPIDSMPF_HAVE_STREAMING)
  add_subdirectory(streaming)
endif()
