/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <vector>

#include <rapidsmpf/bootstrap/types.hpp>

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

/**
 * @brief Get the current `rrun` rank.
 *
 * This helper retrieves the rank of the current process when running with `rrun`.
 * The rank is fetched from the `RAPIDSMPF_RANK` environment variable.
 *
 * @return Rank of the current process (>= 0) if found, -1 otherwise.
 */
Rank get_rank();

/**
 * @brief Get the number of `rrun` ranks.
 *
 * This helper retrieves the number of ranks when running with `rrun`.
 * The number of ranks is fetched from the `RAPIDSMPF_NRANKS` environment variable.
 *
 * @return Number of ranks.
 * @throws std::runtime_error if not running with `rrun` or if `RAPIDSMPF_NRANKS` is not
 * set or cannot be parsed.
 */
Rank get_nranks();

/**
 * @brief Parse CPU list string into vector of core IDs.
 *
 * Parses a comma-separated CPU list string that may contain ranges (e.g., "0-3,8-11")
 * into a vector of individual CPU core IDs.
 *
 * @param cpulist CPU list string (e.g., "0-3,8-11" or "0,1,2,3").
 * @return Vector of CPU core IDs. Empty if parsing fails or input is empty.
 */
std::vector<int> parse_cpu_list(std::string const& cpulist);

/**
 * @brief Compare two CPU affinity strings (order-independent).
 *
 * Compares two CPU affinity strings by parsing them into sorted lists of core IDs
 * and checking if they contain the same cores, regardless of order or formatting.
 *
 * @param actual Actual CPU affinity string.
 * @param expected Expected CPU affinity string.
 * @return true if both strings represent the same set of CPU cores, false otherwise.
 */
bool compare_cpu_affinity(std::string const& actual, std::string const& expected);

/**
 * @brief Compare two comma-separated device lists (order-independent).
 *
 * Compares two comma-separated device lists by parsing them into sorted vectors
 * and checking if they contain the same devices, regardless of order.
 *
 * @param actual Actual device list string.
 * @param expected Expected device list string.
 * @return true if both strings represent the same set of devices, false otherwise.
 */
bool compare_device_lists(std::string const& actual, std::string const& expected);

}  // namespace rapidsmpf::bootstrap
