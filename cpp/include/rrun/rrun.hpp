/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cucascade/memory/topology_discovery.hpp>

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
 * @param gpu_id  GPU device index (as reported by `nvidia-smi`) to bind for.
 *                Negative values are silently ignored.
 * @param options Controls which resource bindings to apply.
 */
void bind(int gpu_id, bind_options const& options = {});

/**
 * @brief Bind using pre-discovered topology information.
 *
 * Same as the single-argument overload, but skips the topology discovery step
 * by reusing a previously obtained `system_topology_info`.  Useful when the
 * caller has already performed discovery (e.g., in a parent process before
 * forking).
 *
 * @param gpu_id   GPU device index to bind for.  Negative values are silently
 *                 ignored.
 * @param topology Pre-discovered system topology.
 * @param options  Controls which resource bindings to apply.
 */
void bind(
    int gpu_id,
    cucascade::memory::system_topology_info const& topology,
    bind_options const& options = {}
);

}  // namespace rapidsmpf::rrun
