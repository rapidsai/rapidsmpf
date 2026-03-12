# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =================================================================================

set(cython_modules
    buffer.pyx buffer_resource.pyx content_description.pyx memory_reservation.pyx packed_data.pyx
    pinned_memory_resource.pyx scoped_memory_record.pyx spill_manager.pyx
)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_modules}"
  LINKED_LIBRARIES rapidsmpf::rapidsmpf maybe_asan
)
