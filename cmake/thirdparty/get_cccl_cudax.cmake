# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =================================================================================

# This function finds CCCL with CUDAX.
function(find_and_configure_cccl_cudax)
  message(STATUS "Finding CCCL with CUDAX")
  include("${rapids-cmake-dir}/cpm/cccl.cmake")
  rapids_cpm_cccl(
    BUILD_EXPORT_SET rapidsmpf-exports
    INSTALL_EXPORT_SET rapidsmpf-exports
    ENABLE_UNSTABLE
  )
endfunction()

find_and_configure_cccl_cudax()
