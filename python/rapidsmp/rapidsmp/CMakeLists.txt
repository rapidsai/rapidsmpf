# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =================================================================================

set(cython_modules exception_handling.pyx progress_thread.pyx shuffler.pyx statistics.pyx)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_modules}"
  LINKED_LIBRARIES rapidsmpf::rapidsmp MPI::MPI_CXX ucxx::ucxx ucx::ucp
)

add_subdirectory(communicator)
add_subdirectory(buffer)
