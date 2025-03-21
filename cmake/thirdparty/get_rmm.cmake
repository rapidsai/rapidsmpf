# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =================================================================================

# This function finds rmm and sets any additional necessary environment variables.
function(find_and_configure_rmm)
  include(${rapids-cmake-dir}/cpm/rmm.cmake)
  rapids_cpm_rmm(BUILD_EXPORT_SET rapidsmp-exports INSTALL_EXPORT_SET rapidsmp-exports)
endfunction()

find_and_configure_rmm()
