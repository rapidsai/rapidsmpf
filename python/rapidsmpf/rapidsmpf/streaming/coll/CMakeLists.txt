# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =================================================================================

set(cython_modules collective.pyx allgather.pyx shuffler.pyx sparse_alltoall.pyx)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_modules}"
  LINKED_LIBRARIES rapidsmpf::rapidsmpf maybe_asan MODULE_PREFIX streaming_coll_
)
