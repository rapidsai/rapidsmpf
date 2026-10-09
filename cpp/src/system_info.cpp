/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <algorithm>
#include <iterator>
#include <vector>

#include <sched.h>
#include <unistd.h>

#include <cucascade/memory/topology_discovery.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/system_info.hpp>


#if RAPIDSMPF_HAVE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

namespace rapidsmpf {

#if RAPIDSMPF_HAVE_NUMA
namespace {
/**
 * @brief Convert a NUMA bitmask to a list of node IDs.
 *
 * @param mask The bitmask. A null mask yields an empty list.
 * @param pred Only nodes for which `pred(node)` is true are returned.
 */
std::vector<int> bitmask_to_nodes(struct bitmask const* mask, auto pred) {
    std::vector<int> ret;
    if (mask == nullptr) {
        return ret;
    }

    for (unsigned int node = 0; node < mask->size; ++node) {
        if (numa_bitmask_isbitset(mask, node) != 0 && pred(static_cast<int>(node))) {
            ret.push_back(static_cast<int>(node));
        }
    }
    return ret;
}
}  // namespace
#endif

std::uint64_t get_total_host_memory() noexcept {
    static const std::uint64_t ret = [] {
        auto const page_size = ::sysconf(_SC_PAGE_SIZE);
        auto const phys_pages = ::sysconf(_SC_PHYS_PAGES);
        RAPIDSMPF_EXPECTS_FATAL(
            page_size != -1 && phys_pages != -1,
            "get_total_host_memory() - fatal error: "
            "sysconf(_SC_PAGE_SIZE/_SC_PHYS_PAGES) failed"
        );
        return safe_cast<std::uint64_t>(page_size) * safe_cast<std::uint64_t>(phys_pages);
    }();
    return ret;
}

int get_current_numa_node() noexcept {
#if RAPIDSMPF_HAVE_NUMA
    static const int ret = [] {
        if (numa_available() == -1) {
            return 0;
        }
        return numa_node_of_cpu(sched_getcpu());
    }();
    return ret;
#else
    return 0;
#endif
}

std::vector<int> get_allowed_host_numa_nodes() noexcept {
#if RAPIDSMPF_HAVE_NUMA
    if (numa_available() == -1) {
        return {0};
    }
    std::vector<int> ret;
    struct bitmask* allowed = numa_allocate_nodemask();
    struct bitmask* cpus = numa_allocate_cpumask();
    if (allowed != nullptr && cpus != nullptr
        && get_mempolicy(
               nullptr, allowed->maskp, allowed->size, nullptr, MPOL_F_MEMS_ALLOWED
           ) == 0)
    {
        // Filter out nodes that have no CPUs (e.g., GPU HBM nodes).
        ret = bitmask_to_nodes(allowed, [&](int node) {
            return numa_node_to_cpus(node, cpus) == 0 && numa_bitmask_weight(cpus) > 0;
        });
    }
    numa_free_nodemask(allowed);
    numa_free_cpumask(cpus);
    if (ret.empty()) {
        return {0};
    }
    return ret;
#else
    return {0};
#endif
}

std::vector<int> get_current_numa_nodes() noexcept {
#if RAPIDSMPF_HAVE_NUMA
    if (numa_available() == -1) {
        return {0};
    }

    // Host-memory nodes the process may allocate from.
    auto const host_nodes = get_allowed_host_numa_nodes();
    // Policy nodes that are valid host nodes.
    std::vector<int> policy_host_nodes;
    int mode = MPOL_DEFAULT;
    struct bitmask* policy = numa_allocate_nodemask();
    if (policy != nullptr
        && get_mempolicy(&mode, policy->maskp, policy->size, nullptr, 0) == 0)
    {
        policy_host_nodes = bitmask_to_nodes(policy, [&](int node) {
            return std::ranges::find(host_nodes, node) != host_nodes.end();
        });
    }
    numa_free_nodemask(policy);  // no-op on nullptr
    // Policy host nodes, followed by the remaining host nodes.
    auto const policy_first = [&] {
        auto ret = policy_host_nodes;
        std::ranges::copy_if(host_nodes, std::back_inserter(ret), [&](int node) {
            return std::ranges::find(policy_host_nodes, node) == policy_host_nodes.end();
        });
        return ret;
    };

    // Fall back to node 0 if no valid host node was found.
    auto const or_node_zero = [](std::vector<int> nodes) {
        return nodes.empty() ? std::vector<int>{0} : nodes;
    };
    // Strip mode flags (MPOL_F_STATIC_NODES, MPOL_F_RELATIVE_NODES, ...).
    switch (mode & ~0xE000) {
    case MPOL_BIND:
        // Allocation is restricted to the bound nodes.
        return or_node_zero(policy_host_nodes);
    case MPOL_PREFERRED:
    case MPOL_INTERLEAVE:
#ifdef MPOL_PREFERRED_MANY
    case MPOL_PREFERRED_MANY:
#endif
#ifdef MPOL_WEIGHTED_INTERLEAVE
    case MPOL_WEIGHTED_INTERLEAVE:
#endif
        // Allocation may fall back to any allowed node; list policy nodes first.
        return or_node_zero(policy_first());
    default:  // MPOL_DEFAULT, MPOL_LOCAL
        return or_node_zero(host_nodes);
    }
#else
    return {0};
#endif
}

std::uint64_t get_numa_node_host_memory([[maybe_unused]] int numa_id) noexcept {
    long long ret = -1;

#if RAPIDSMPF_HAVE_NUMA
    if (numa_available() == -1) {
        return get_total_host_memory();
    }
    long long ignored = 0;
    ret = numa_node_size64(numa_id, &ignored);
#endif

    if (ret == -1) {
        return get_total_host_memory();
    }
    return safe_cast<std::uint64_t>(ret);
}

namespace {
const auto& get_topology() {
    static const auto topo = [] {
        cucascade::memory::topology_discovery discovery;
        RAPIDSMPF_EXPECTS(
            discovery.discover(), "Failed to discover system topology", std::runtime_error
        );
        return discovery;
    }();
    return topo.get_topology();
}
}  // namespace

std::uint64_t get_host_memory_per_gpu() {
    auto const current_numa_node = get_current_numa_node();
    auto const& gpus = get_topology().gpus;
    // gpu.numa_node == -1 means the kernel has no NUMA affinity info for the
    // device (common in VMs and single-socket machines without ACPI SRAT/SLIT
    // entries for PCIe).  Treat those GPUs as local to every NUMA node.
    auto const num_local_gpus = std::ranges::count_if(gpus, [&](auto const& gpu) {
        return gpu.numa_node == current_numa_node || gpu.numa_node == -1;
    });
    RAPIDSMPF_EXPECTS(
        num_local_gpus > 0,
        "No GPUs found on current NUMA node " + std::to_string(current_numa_node),
        std::runtime_error
    );
    return get_numa_node_host_memory(current_numa_node)
           / safe_cast<std::uint64_t>(num_local_gpus);
}

}  // namespace rapidsmpf
