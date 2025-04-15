# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =================================================================================

# This function finds gtest and sets any additional necessary environment variables.
function(find_and_configure_gtest)
  include(${rapids-cmake-dir}/cpm/gtest.cmake)

  # Find or install GoogleTest
  rapids_cpm_gtest(
    BUILD_EXPORT_SET rapidsmpf-testing-exports INSTALL_EXPORT_SET rapidsmpf-testing-exports
  )

  if(GTest_ADDED)
    rapids_export(
      BUILD GTest
      VERSION ${GTest_VERSION}
      EXPORT_SET GTestTargets
      GLOBAL_TARGETS gtest gmock gtest_main gmock_main
      NAMESPACE GTest::
    )

    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(
      BUILD GTest [=[${CMAKE_CURRENT_LIST_DIR}]=] EXPORT_SET rapidsmpf-testing-exports
    )
  endif()

endfunction()

find_and_configure_gtest()
