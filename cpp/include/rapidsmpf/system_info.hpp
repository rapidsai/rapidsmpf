/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace rapidsmpf {

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

}  // namespace rapidsmpf
