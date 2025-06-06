# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =================================================================================

set(cython_modules config.pyx progress_thread.pyx shuffler.pyx statistics.pyx)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_modules}"
  LINKED_LIBRARIES rapidsmpf::rapidsmpf
)

add_subdirectory(_detail)
add_subdirectory(buffer)
add_subdirectory(communicator)
add_subdirectory(utils)
