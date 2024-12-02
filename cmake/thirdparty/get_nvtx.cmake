# =============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# Need to call rapids_cpm_nvtx3 to get support for an installed version of nvtx3 and to support
# installing it ourselves
function(find_and_configure_nvtx)
  include(${rapids-cmake-dir}/cpm/nvtx3.cmake)

  # Find or install nvtx3
  rapids_cpm_nvtx3(BUILD_EXPORT_SET kvikio-exports INSTALL_EXPORT_SET kvikio-exports)

endfunction()

find_and_configure_nvtx()
