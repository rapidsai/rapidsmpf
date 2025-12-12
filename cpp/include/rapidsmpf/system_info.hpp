/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace rapidsmpf {

/**
 * @brief Get the total amount of host (main) memory.
 *
 * @return Total host memory in bytes.
 *
 * @note On WSL and in containerized environments, the returned value
 * reflects the memory visible to the Linux kernel instance, which may
 * differ from the host system's physical memory.
 */
std::uint64_t get_total_host_memory() noexcept;

/**
 * @brief Get the NUMA node ID associated with the calling CPU thread.
 *
 * A NUMA (Non-Uniform Memory Access) node represents a group of CPU cores and
 * memory that have faster access to each other than to memory attached to
 * other nodes. On NUMA systems, binding allocations and threads to the same
 * NUMA node can significantly reduce memory access latency and improve
 * bandwidth.
 *
 * This function returns the NUMA node on which the calling thread is currently
 * executing, as determined by the operating system's CPU and memory topology.
 * The value can change if the thread migrates between CPUs.
 *
 * If NUMA support is not available on the system or cannot be queried, the
 * function returns 0, which corresponds to the single implicit NUMA node on
 * non-NUMA systems.
 *
 * @return The NUMA node ID of the calling thread, or 0 if NUMA is unavailable.
 */
int get_current_numa_node_id() noexcept;

/**
 * @brief Get the total amount of host memory for a NUMA node.
 *
 * @param numa_id
 *     NUMA node for which to query the total host memory. Defaults to the
 *     current NUMA node as returned by `get_current_numa_node_id()`.
 *
 * @note If NUMA support is not available or the node size cannot be
 * determined, this function falls back to returning the total host memory.
 *
 * @return Total host memory of the NUMA node in bytes.
 */
std::uint64_t get_numa_node_host_memory(
    int numa_id = get_current_numa_node_id()
) noexcept;


}  // namespace rapidsmpf
