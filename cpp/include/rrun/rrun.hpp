/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

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
    bool verify{true};  ///< Read back and verify bindings after applying them.
};

/**
 * @brief Live resource binding configuration collected from the running process.
 *
 * Holds the CPU affinity, NUMA memory binding, and network device configuration
 * that are currently in effect. Obtained via `check_binding()`.
 */
struct resource_binding {
    int rank = -1;  ///< Process rank (-1 if not available).
    int gpu_id = -1;  ///< GPU device ID (-1 if not available).
    std::string gpu_pci_bus_id;  ///< GPU PCI bus ID (empty if unavailable).
    std::string cpu_affinity;  ///< CPU affinity string (e.g., "0-19,40-59").
    std::vector<int> numa_nodes;  ///< NUMA node IDs bound to this process.
    std::string ucx_net_devices;  ///< Value of the `UCX_NET_DEVICES` env var.
};

/**
 * @brief Expected resource binding derived from topology information.
 *
 * Represents the binding configuration that *should* be in effect for a given
 * GPU according to the system topology.
 */
struct expected_binding {
    std::string cpu_affinity;  ///< Expected CPU affinity list.
    std::vector<int> memory_binding;  ///< Expected NUMA node IDs.
    std::vector<std::string> network_devices;  ///< Expected network devices.
};

/**
 * @brief Results of validating actual vs. expected resource bindings.
 */
struct binding_validation {
    bool cpu_ok = true;  ///< CPU affinity check passed.
    bool numa_ok = true;  ///< NUMA binding check passed.
    bool ucx_ok = true;  ///< UCX network devices check passed.
    std::string expected_ucx_devices;  ///< Expected UCX devices (comma-separated).

    /**
     * @brief Check if all validations passed.
     * @return true if CPU, NUMA, and UCX checks all passed.
     */
    [[nodiscard]] bool all_passed() const {
        return cpu_ok && numa_ok && ucx_ok;
    }
};

/**
 * @brief Collect the live resource binding of the calling process.
 *
 * Queries the current CPU affinity, NUMA memory nodes, UCX network device
 * configuration, process rank, and GPU information. Fields that cannot be
 * determined (e.g. rank when no launcher environment is set, or GPU ID when
 * `CUDA_VISIBLE_DEVICES` is absent and no hint is given) are left at their
 * default value of -1.
 *
 * @param gpu_id_hint GPU device index hint. When >= 0 the value is stored
 * directly; otherwise the GPU ID is read from `CUDA_VISIBLE_DEVICES`.
 * When a valid GPU ID is available, the PCI bus ID is also queried.
 *
 * @return The collected resource binding.
 */
resource_binding check_binding(int gpu_id_hint = -1);

/**
 * @brief Obtain the expected binding for a GPU from pre-discovered topology.
 *
 * Looks up @p gpu_id in @p topology and returns the expected CPU affinity,
 * memory binding, and network devices.
 *
 * @param topology Pre-discovered system topology.
 * @param gpu_id GPU device index to look up.
 * @return The expected binding, or `std::nullopt` if @p gpu_id is not found.
 */
std::optional<expected_binding> get_expected_binding(
    cucascade::memory::system_topology_info const& topology, int gpu_id
);

/**
 * @brief Validate an actual resource binding against an expected one.
 *
 * Compares the live @p actual binding with @p expected and reports per-resource
 * pass/fail status.
 *
 * @param actual Live resource binding (from `check_binding()`).
 * @param expected Expected binding (from topology or a JSON file).
 * @return Validation results.
 */
binding_validation validate_binding(
    resource_binding const& actual, expected_binding const& expected
);

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
 * @throws std::runtime_error if no GPU ID can be determined, topology
 * discovery fails, the resolved GPU is not found in the discovered topology,
 * an enabled binding (CPU affinity, NUMA memory policy, network devices) could
 * not be applied, or post-bind verification detects a mismatch between the
 * requested and actual binding state.
 */
void bind(
    std::optional<unsigned int> gpu_id = std::nullopt, bind_options const& options = {}
);

/**
 * @brief Bind using pre-discovered topology information.
 *
 * Same as the other overload, but skips the topology discovery step by reusing
 * a previously obtained `system_topology_info`. Useful when the caller has
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
 * @throws std::runtime_error if no GPU ID can be determined, the resolved GPU
 * is not found in @p topology, an enabled binding (CPU affinity, NUMA memory
 * policy, network devices) could not be applied, or post-bind verification
 * detects a mismatch between the requested and actual binding state.
 */
void bind(
    cucascade::memory::system_topology_info const& topology,
    std::optional<unsigned int> gpu_id = std::nullopt,
    bind_options const& options = {}
);

}  // namespace rapidsmpf::rrun
