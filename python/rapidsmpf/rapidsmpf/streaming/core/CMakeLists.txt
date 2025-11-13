# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =================================================================================

set(cython_modules channel.pyx context.pyx fanout.pyx leaf_node.pyx message.pyx
                   spillable_messages.pyx node.pyx utilities.pyx
)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_modules}"
  LINKED_LIBRARIES rapidsmpf::rapidsmpf maybe_asan MODULE_PREFIX streaming_core_
)
