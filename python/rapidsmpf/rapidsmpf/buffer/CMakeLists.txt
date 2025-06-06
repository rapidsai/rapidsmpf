# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =================================================================================

set(cython_modules buffer.pyx packed_data.pyx resource.pyx spill_manager.pyx
                   rmm_fallback_resource.pyx
)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_modules}"
  LINKED_LIBRARIES rapidsmpf::rapidsmpf ucxx::ucxx ucx::ucp $<TARGET_NAME_IF_EXISTS:MPI::MPI_C>
)
