/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime_api.h>

namespace rapidsmpf {

/** @brief Helper macro to check if the CUDA version is at least the specified version.
 *
 * @param version The minimum CUDA version to check against. Must be in the format of
 * MAJOR*1000 + MINOR*10.
 */
#define RAPIDSMPF_CUDA_VERSION_AT_LEAST(version) (CUDART_VERSION >= version)

/**
 * @brief Gets the NUMA node ID of the current CPU process.
 *
 * @note This function is only available if built with NUMA support. (See
 * `RAPIDSMPF_NUMA_SUPPORT` CMake option.)
 *
 * @return The NUMA node ID of the current CPU process.
 *
 * @throws std::runtime_error If built with NUMA support but libnuma is not available
 * at runtime or if the NUMA node ID cannot be retrieved.
 */
int get_current_numa_node_id();

}  // namespace rapidsmpf
