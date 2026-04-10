/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <sched.h>
#include <unistd.h>

#if RAPIDSMPF_HAVE_NUMA
#include <numa.h>
#endif

#include <cucascade/memory/topology_discovery.hpp>

#include <rrun/rrun.hpp>

namespace rapidsmpf::rrun {

namespace {

/**
 * @brief Parse a CPU list string (e.g. "0-31,128-159") into a cpu_set_t mask.
 *
 * @param cpulist CPU list string.
 * @param cpuset  Output CPU set to populate.
 * @return true on success, false on failure.
 */
bool parse_cpu_list_to_mask(std::string const& cpulist, cpu_set_t* cpuset) {
    CPU_ZERO(cpuset);
    if (cpulist.empty()) {
        return false;
    }

    std::istringstream iss(cpulist);
    std::string token;
    while (std::getline(iss, token, ',')) {
        std::size_t dash_pos = token.find('-');
        if (dash_pos != std::string::npos) {
            try {
                int start = std::stoi(token.substr(0, dash_pos));
                int end = std::stoi(token.substr(dash_pos + 1));
                for (int i = start; i <= end; ++i) {
                    if (i >= 0 && i < static_cast<int>(CPU_SETSIZE)) {
                        CPU_SET(static_cast<unsigned>(i), cpuset);
                    }
                }
            } catch (...) {
                return false;
            }
        } else {
            try {
                int core = std::stoi(token);
                if (core >= 0 && core < static_cast<int>(CPU_SETSIZE)) {
                    CPU_SET(static_cast<unsigned>(core), cpuset);
                }
            } catch (...) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Set CPU affinity for the current process.
 *
 * @param cpu_affinity_list CPU affinity list string (e.g., "0-31,128-159"), as in the
 * format of `cucascade::memory::gpu_topology_info::cpu_affinity_list`.
 * @return true on success, false on failure.
 */
bool set_cpu_affinity(std::string const& cpu_affinity_list) {
    if (cpu_affinity_list.empty()) {
        return false;
    }

    cpu_set_t cpuset;
    if (!parse_cpu_list_to_mask(cpu_affinity_list, &cpuset)) {
        return false;
    }

    pid_t pid = getpid();
    if (sched_setaffinity(pid, sizeof(cpu_set_t), &cpuset) != 0) {
        return false;
    }

    return true;
}

/**
 * @brief Set NUMA memory binding for the current process.
 *
 * @param memory_binding Vector of NUMA node IDs to bind memory to.
 * @return true on success, false on failure or if NUMA is not available.
 */
bool set_numa_memory_binding(std::vector<int> const& memory_binding) {
#if RAPIDSMPF_HAVE_NUMA
    if (memory_binding.empty()) {
        return false;
    }

    if (numa_available() == -1) {
        return false;
    }

    struct bitmask* nodemask = numa_allocate_nodemask();
    if (!nodemask) {
        return false;
    }

    numa_bitmask_clearall(nodemask);
    for (int node : memory_binding) {
        if (node >= 0) {
            numa_bitmask_setbit(nodemask, static_cast<unsigned int>(node));
        }
    }

    numa_set_membind(nodemask);
    numa_free_nodemask(nodemask);

    return true;
#else
    std::ignore = memory_binding;
    return false;
#endif
}

/**
 * @brief Apply topology-based resource bindings for a single GPU.
 *
 * @param gpu_info Topology information for the target GPU.
 * @param gpu_id   GPU device index (used only in diagnostic messages).
 * @param options  Which bindings to apply.
 */
void apply_bindings(
    cucascade::memory::gpu_topology_info const& gpu_info,
    unsigned int gpu_id,
    bind_options const& options
) {
    if (options.cpu && !gpu_info.cpu_affinity_list.empty()) {
        if (!set_cpu_affinity(gpu_info.cpu_affinity_list)) {
            if (options.verbose) {
                std::cerr << "[rrun] Warning: Failed to set CPU affinity for GPU "
                          << gpu_id << std::endl;
            }
        }
    }

    if (options.memory && !gpu_info.memory_binding.empty()) {
        if (!set_numa_memory_binding(gpu_info.memory_binding)) {
#if RAPIDSMPF_HAVE_NUMA
            if (options.verbose) {
                std::cerr << "[rrun] Warning: Failed to set NUMA memory binding for GPU "
                          << gpu_id << std::endl;
            }
#endif
        }
    }

    if (options.network && !gpu_info.network_devices.empty()) {
        std::string ucx_net_devices;
        for (std::size_t i = 0; i < gpu_info.network_devices.size(); ++i) {
            if (i > 0) {
                ucx_net_devices += ",";
            }
            ucx_net_devices += gpu_info.network_devices[i];
        }
        setenv("UCX_NET_DEVICES", ucx_net_devices.c_str(), 1);
    }
}

/**
 * @brief Resolve a GPU ID from an explicit value or `CUDA_VISIBLE_DEVICES`.
 *
 * @param gpu_id Caller-supplied GPU ID, or `std::nullopt` to fall back.
 * @return Resolved GPU device index.
 * @throws std::runtime_error if no GPU ID can be determined.
 */
unsigned int resolve_gpu_id(std::optional<unsigned int> gpu_id) {
    if (gpu_id.has_value()) {
        return *gpu_id;
    }

    char const* env = std::getenv("CUDA_VISIBLE_DEVICES");
    if (env != nullptr && env[0] != '\0') {
        std::string first_entry(env);
        auto comma = first_entry.find(',');
        if (comma != std::string::npos) {
            first_entry.resize(comma);
        }
        try {
            int value = std::stoi(first_entry);
            if (value < 0) {
                throw std::out_of_range("negative");
            }
            return static_cast<unsigned int>(value);
        } catch (...) {
            throw std::runtime_error(
                "rapidsmpf::rrun::bind(): CUDA_VISIBLE_DEVICES first entry ('"
                + first_entry + "') is not a valid GPU ID"
            );
        }
    }

    throw std::runtime_error(
        "rapidsmpf::rrun::bind(): no GPU ID specified and CUDA_VISIBLE_DEVICES "
        "is not set"
    );
}

}  // namespace

void bind(std::optional<unsigned int> gpu_id, bind_options const& options) {
    unsigned int id = resolve_gpu_id(gpu_id);

    // Temporarily clear CUDA_VISIBLE_DEVICES so the topology discovery layer
    // sees all physical GPUs.  When the variable restricts visibility to a
    // single device, NVML remaps it to index 0 and a lookup by the real
    // physical ID would fail.
    char const* cvd = std::getenv("CUDA_VISIBLE_DEVICES");
    std::string saved_cvd;
    bool had_cvd = false;
    if (cvd != nullptr) {
        had_cvd = true;
        saved_cvd = cvd;
        unsetenv("CUDA_VISIBLE_DEVICES");
    }

    cucascade::memory::topology_discovery discovery;
    bool ok = discovery.discover();

    if (had_cvd) {
        setenv("CUDA_VISIBLE_DEVICES", saved_cvd.c_str(), 1);
    }

    if (!ok) {
        if (options.verbose) {
            std::cerr << "[rrun] Warning: Failed to discover system topology. "
                      << "CPU affinity, NUMA binding, and UCX network device "
                      << "configuration will be skipped." << std::endl;
        }
        return;
    }
    bind(discovery.get_topology(), id, options);
}

void bind(
    cucascade::memory::system_topology_info const& topology,
    std::optional<unsigned int> gpu_id,
    bind_options const& options
) {
    unsigned int id = resolve_gpu_id(gpu_id);

    for (auto const& gpu : topology.gpus) {
        if (gpu.id == id) {
            apply_bindings(gpu, id, options);
            return;
        }
    }

    throw std::runtime_error(
        "rapidsmpf::rrun::bind(): GPU " + std::to_string(id) + " not found in topology"
    );
}

}  // namespace rapidsmpf::rrun
