/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <vector>

#include <rapidsmpf/config.hpp>

namespace rapidsmpf::bootstrap {

/**
 * @brief Get current CPU affinity as a string.
 *
 * Queries the current process's CPU affinity mask and formats it as a
 * comma-separated list of CPU core IDs, with ranges represented as "start-end".
 *
 * Example output: "0-3,8-11" for cores 0,1,2,3,8,9,10,11
 *
 * @return CPU affinity string, or empty string on error.
 */
std::string get_current_cpu_affinity();

/**
 * @brief Get current NUMA node(s) for memory binding.
 *
 * Queries the NUMA node associated with the CPU the current process is running on.
 * This is a best-effort approach and may not be accurate in all cases.
 *
 * @return Vector of NUMA node IDs. Empty if NUMA is not available or detection fails.
 */
std::vector<int> get_current_numa_nodes();

/**
 * @brief Get UCX_NET_DEVICES from environment.
 *
 * Retrieves the value of the UCX_NET_DEVICES environment variable, which
 * specifies which network devices UCX should use for communication.
 *
 * @return Value of UCX_NET_DEVICES, or empty string if not set.
 */
std::string get_ucx_net_devices();

/**
 * @brief Get GPU ID from CUDA_VISIBLE_DEVICES.
 *
 * Attempts to determine the GPU ID assigned to this process by checking the
 * CUDA_VISIBLE_DEVICES environment variable.
 *
 * @return GPU ID (>= 0) if found, -1 otherwise.
 */
int get_gpu_id();

/**
 * @brief Check if the current process was launched via `rrun`.
 *
 * This helper detects bootstrap mode by checking for the presence of the
 * `RAPIDSMPF_RANK` environment variable, which is set by `rrun`.
 *
 * @return true if running under `rrun` bootstrap mode, false otherwise.
 */
bool is_running_with_rrun();

}  // namespace rapidsmpf::bootstrap
