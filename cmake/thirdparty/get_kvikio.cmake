# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds KvikIO
function(find_and_configure_kvikio VERSION)

  rapids_cpm_find(
    kvikio ${VERSION}
    GLOBAL_TARGETS kvikio::kvikio
    CPM_ARGS
    GIT_REPOSITORY https://github.com/rapidsai/kvikio.git
    GIT_TAG main
    GIT_SHALLOW TRUE SOURCE_SUBDIR cpp
    OPTIONS "KvikIO_BUILD_EXAMPLES OFF" "KvikIO_REMOTE_SUPPORT ${CUDF_KVIKIO_REMOTE_IO}"
  )

  include("${rapids-cmake-dir}/export/find_package_root.cmake")
  rapids_export_find_package_root(
    BUILD KvikIO "${KvikIO_BINARY_DIR}"
    EXPORT_SET rapidsmpf-exports
    CONDITION KvikIO_BINARY_DIR
  )

endfunction()

find_and_configure_kvikio(${RAPIDS_VERSION})
