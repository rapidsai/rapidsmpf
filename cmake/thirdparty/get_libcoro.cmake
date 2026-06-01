# ============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# ============================================================================

# This function finds libcoro and sets any additional necessary environment variables.
function(find_and_configure_libcoro)
  include(${rapids-cmake-dir}/cpm/libcoro.cmake)
  rapids_cpm_libcoro(BUILD_EXPORT_SET rapidsmpf-exports BUILD_STATIC)
endfunction()

find_and_configure_libcoro()
