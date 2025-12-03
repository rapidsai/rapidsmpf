# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =================================================================================

set(modules_need_ucxx bootstrap.pyx)

if(RAPIDSMPF_HAVE_UCXX)
  rapids_cython_create_modules(
    CXX
    SOURCE_FILES "${modules_need_ucxx}"
    LINKED_LIBRARIES rapidsmpf::rapidsmpf maybe_asan
  )
endif()
