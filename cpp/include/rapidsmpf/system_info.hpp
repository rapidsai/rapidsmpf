/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <vector>

namespace rapidsmpf {

/**
 * @brief Get the total amount of system memory.
 *
 * @return Total host memory in bytes.
 *
 * @note On WSL and in containerized environments, the returned value
 * reflects the memory visible to the Linux kernel instance, which may
 * differ from the physical memory of the host.
 *
 * @note Terminates the process if `sysconf(_SC_PAGE_SIZE)` or
 * `sysconf(_SC_PHYS_PAGES)` fails.
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
int get_current_numa_node() noexcept;

/**
 * @brief Get the host-memory NUMA nodes the process is allowed to allocate from.
 *
 * Unlike `get_current_numa_nodes()`, this ignores the memory policy mode and reports
 * every node the process may use (respecting cpuset restrictions, e.g. in containers).
 * Only nodes that have CPUs are returned, which excludes GPU HBM nodes exposed on GB/GH
 * systems (CPU-less CXL expander nodes are also excluded). Nodes are in ascending order.
 *
 * If NUMA support is not available or no node can be determined, the function returns a
 * vector containing a single element, `0`.
 *
 * @return Vector of allowed host-memory NUMA node IDs.
 */
std::vector<int> get_allowed_host_numa_nodes() noexcept;

/**
 * @brief Get current NUMA node(s) for memory binding.
 *
 * Inspects the process memory policy and returns the host-memory NUMA nodes from which
 * the process may allocate. Only nodes that have CPUs are considered, which excludes
 * GPU HBM nodes exposed on GB/GH systems (CPU-less CXL expander nodes are also
 * excluded). The result depends on the policy mode:
 * - `MPOL_DEFAULT`/`MPOL_LOCAL`: all allowed host nodes (respecting cpusets).
 * - `MPOL_BIND`: only the bound nodes. Throws if any bound node is not an allowed host
 *   node (e.g. a CPU-less GPU memory node), rather than silently dropping it.
 * - `MPOL_PREFERRED`, `MPOL_PREFERRED_MANY`: all allowed host nodes, with the preferred
 *   node(s) first.
 * - `MPOL_INTERLEAVE`, `MPOL_WEIGHTED_INTERLEAVE`: all allowed host nodes, with the
 *   interleave nodes first.
 *
 * The memory policy is per-thread in Linux, so this reads the policy of the calling
 * thread, not of the process as a whole. A policy set in one thread (e.g. via
 * `numa_set_membind()`) is not seen when called from another thread.
 *
 * If NUMA support is not available or the NUMA nodes cannot be determined, the
 * function returns a vector containing a single element, `0`, which corresponds
 * to the single implicit NUMA node on non-NUMA systems.
 *
 * Policies using `MPOL_F_RELATIVE_NODES` are treated as the default policy, since
 * their node mask is relative and can't be interpreted as absolute node IDs.
 *
 * @return Vector of NUMA node IDs in the current memory policy.
 *
 * @throws std::runtime_error If the policy is `MPOL_BIND` and includes nodes that are
 * not allowed host-memory nodes.
 */
std::vector<int> get_current_numa_nodes();

/**
 * @brief Get the total amount of host memory for a NUMA node.
 *
 * @param numa_id
 *     NUMA node for which to query the total host memory. Defaults to the
 *     current NUMA node as returned by `get_current_numa_node()`.
 *
 * @note If NUMA support is not available or the node size cannot be
 * determined, this function falls back to returning the total host memory.
 *
 * @return Total host memory of the NUMA node in bytes.
 */
std::uint64_t get_numa_node_host_memory(int numa_id = get_current_numa_node()) noexcept;

/**
 * @brief Get the amount of host memory per GPU.
 *
 * This is calculated as the total host memory available for the current NUMA node divided
 * by the number of visible GPUs bound to that NUMA node. GPUs with unknown NUMA affinity
 * count as local. If there are no local GPUs, it is divided by the number of visible GPUs
 * across all nodes instead. The result is computed once and cached.
 *
 * @throws std::runtime_error if no GPUs are found in the topology.
 *
 * @return Amount of host memory per GPU in bytes.
 */
std::uint64_t get_host_memory_per_gpu();

}  // namespace rapidsmpf
