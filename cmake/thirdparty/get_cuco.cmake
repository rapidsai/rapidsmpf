# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds cuco and sets any additional necessary environment variables.
function(find_and_configure_cucollections)
  include(${rapids-cmake-dir}/cpm/cuco.cmake)

  rapids_cpm_cuco(BUILD_EXPORT_SET rapidsmpf-exports INSTALL_EXPORT_SET rapidsmpf-exports)
endfunction()

find_and_configure_cucollections()
