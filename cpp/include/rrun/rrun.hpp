/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>

namespace cucascade::memory {
struct system_topology_info;
}  // namespace cucascade::memory

namespace rapidsmpf::rrun {

/**
 * @brief Options controlling which topology-based resource bindings to apply.
 *
 * By default all bindings are enabled. Pass a custom instance to `bind()` to
 * selectively enable or disable individual resource classes.
 */
struct bind_options {
    bool cpu{true};  ///< Set CPU affinity to cores near the GPU.
    bool memory{true};  ///< Set NUMA memory policy to nodes near the GPU.
    bool network{true};  ///< Set `UCX_NET_DEVICES` to NICs near the GPU.
    bool verbose{false};  ///< Print warnings to stderr on binding failures.
};

/**
 * @brief Bind the calling process to resources topologically close to a GPU.
 *
 * Discovers the system topology via `cucascade::memory::topology_discovery`,
 * then applies CPU affinity, NUMA memory binding, and/or network device
 * configuration as requested in @p options.
 *
 * This is the self-contained entry point intended for external libraries that
 * do not launch through the `rrun` CLI.
 *
 * @warning This function is **not thread-safe**. It temporarily modifies the
 * `CUDA_VISIBLE_DEVICES` environment variable during topology discovery and
 * mutates process-wide state (CPU affinity, NUMA memory policy, and the
 * `UCX_NET_DEVICES` environment variable). It should be called exactly once
 * per process, ideally early in initialization and before other threads are
 * spawned.
 *
 * GPU resolution order:
 *   1. Use @p gpu_id if provided.
 *   2. Otherwise, parse the first entry of the `CUDA_VISIBLE_DEVICES`
 *      environment variable.
 *   3. If neither is available, throw `std::runtime_error`.
 *
 * @param gpu_id GPU device index (as reported by `nvidia-smi`) to bind for.
 * When `std::nullopt`, the first GPU in `CUDA_VISIBLE_DEVICES` is used instead.
 * @param options Controls which resource bindings to apply.
 *
 * @throws std::runtime_error if no GPU ID can be determined or the resolved
 *         GPU is not found in the discovered topology.
 */
void bind(
    std::optional<unsigned int> gpu_id = std::nullopt, bind_options const& options = {}
);

/**
 * @brief Bind using pre-discovered topology information.
 *
 * Same as the other overload, but skips the topology discovery step by reusing
 * a previously obtained `system_topology_info`.  Useful when the caller has
 * already performed discovery (e.g., in a parent process before forking).
 *
 * GPU resolution follows the same order as the other overload (explicit
 * @p gpu_id, then `CUDA_VISIBLE_DEVICES`).
 *
 * @warning This function is **not thread-safe**. It mutates process-wide state
 * (CPU affinity, NUMA memory policy, and the `UCX_NET_DEVICES` environment
 * variable). It should be called exactly once per process, ideally early in
 * initialization and before other threads are spawned.
 *
 * @param topology Pre-discovered system topology.
 * @param gpu_id GPU device index to bind for. When `std::nullopt`, the first
 * GPU in `CUDA_VISIBLE_DEVICES` is used instead.
 * @param options Controls which resource bindings to apply.
 *
 * @throws std::runtime_error if no GPU ID can be determined or the resolved
 *         GPU is not found in @p topology.
 */
void bind(
    cucascade::memory::system_topology_info const& topology,
    std::optional<unsigned int> gpu_id = std::nullopt,
    bind_options const& options = {}
);

}  // namespace rapidsmpf::rrun
