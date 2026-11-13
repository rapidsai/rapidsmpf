/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <vector>

#include <sched.h>
#include <unistd.h>

#include <cucascade/memory/topology_discovery.hpp>

#include <rapidsmpf/detail/system_info.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/system_info.hpp>


#if RAPIDSMPF_HAVE_NUMA
#include <numa.h>
#include <numaif.h>

// Older libnuma/kernel headers may not define these.
#ifndef MPOL_F_STATIC_NODES
#define MPOL_F_STATIC_NODES (1 << 15)
#endif
#ifndef MPOL_F_RELATIVE_NODES
#define MPOL_F_RELATIVE_NODES (1 << 14)
#endif
#ifndef MPOL_F_NUMA_BALANCING
#define MPOL_F_NUMA_BALANCING (1 << 13)
#endif
#ifndef MPOL_WEIGHTED_INTERLEAVE
#define MPOL_WEIGHTED_INTERLEAVE 6
#endif
#ifndef MPOL_PREFERRED_MANY
#define MPOL_PREFERRED_MANY 5
#endif
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

std::vector<int> get_current_numa_nodes() {
#if RAPIDSMPF_HAVE_NUMA
    if (numa_available() == -1) {
        return {0};
    }

    // Host-memory nodes the process may allocate from.
    auto const host_nodes = get_allowed_host_numa_nodes();
    // Policy nodes that are valid host nodes.
    std::vector<int> policy_host_nodes;
    // Policy nodes within the cpuset, including non-host ones (e.g. GPU HBM nodes).
    // Nodes outside the cpuset are ignored: with MPOL_F_STATIC_NODES the policy mask
    // is the user-supplied one and may include nodes the kernel never allocates from.
    std::vector<int> bound_nodes;
    int mode = MPOL_DEFAULT;
    struct bitmask* policy = numa_allocate_nodemask();
    struct bitmask* mems_allowed = numa_allocate_nodemask();
    if (policy != nullptr && mems_allowed != nullptr
        && get_mempolicy(&mode, policy->maskp, policy->size, nullptr, 0) == 0
        && get_mempolicy(
               nullptr,
               mems_allowed->maskp,
               mems_allowed->size,
               nullptr,
               MPOL_F_MEMS_ALLOWED
           ) == 0)
    {
        bound_nodes = bitmask_to_nodes(policy, [&](int node) {
            return numa_bitmask_isbitset(mems_allowed, static_cast<unsigned int>(node))
                   != 0;
        });
        policy_host_nodes = bitmask_to_nodes(policy, [&](int node) {
            return std::ranges::find(host_nodes, node) != host_nodes.end();
        });
    }
    numa_free_nodemask(policy);
    numa_free_nodemask(mems_allowed);
    // The policy host nodes first, then every other allowed host node (used by
    // policies that may fall back to any allowed node).
    auto const policy_first = [&] {
        auto ret = policy_host_nodes;
        std::ranges::copy_if(host_nodes, std::back_inserter(ret), [&](int node) {
            return std::ranges::find(policy_host_nodes, node) == policy_host_nodes.end();
        });
        return ret;
    };

    // Strip the mode flags, leaving only the policy mode.
    constexpr int mode_flags =
        MPOL_F_STATIC_NODES | MPOL_F_RELATIVE_NODES | MPOL_F_NUMA_BALANCING;
    // With MPOL_F_RELATIVE_NODES the returned mask is relative to the allowed nodes
    // rather than holding absolute node IDs, so it can't be interpreted here. Treat
    // such policies as the default policy.
    if ((mode & MPOL_F_RELATIVE_NODES) != 0) {
        return host_nodes;
    }
    switch (mode & ~mode_flags) {
    case MPOL_BIND:
        {
            // Allocation is restricted to the bound nodes. Bound nodes that aren't
            // allowed host nodes (e.g. GPU HBM nodes) are an error rather than being
            // silently dropped, since that would report a binding that wasn't requested.
            RAPIDSMPF_EXPECTS(
                !policy_host_nodes.empty()
                    && policy_host_nodes.size() == bound_nodes.size(),
                "MPOL_BIND includes NUMA nodes that are not allowed host-memory nodes "
                "(e.g. CPU-less GPU memory nodes); bind only to host nodes with CPUs",
                std::runtime_error
            );
            return policy_host_nodes;
        }
    case MPOL_PREFERRED:
    case MPOL_INTERLEAVE:
    case MPOL_PREFERRED_MANY:
    case MPOL_WEIGHTED_INTERLEAVE:
        // Allocation may fall back to any allowed node; list policy nodes first.
        return policy_first();
    default:  // MPOL_DEFAULT, MPOL_LOCAL
        return host_nodes;
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
    static const std::uint64_t ret = [] {
        auto const current_numa_node = get_current_numa_node();
        auto const& gpus = get_topology().gpus;
        // gpu.numa_node == -1 means the kernel has no NUMA affinity info for the
        // device (common in VMs and single-socket machines without ACPI SRAT/SLIT
        // entries for PCIe). Those GPUs count as local to every NUMA node. If none
        // are local, an unbound process can still use GPUs on other NUMA nodes.
        auto const gpu_count = detail::gpu_count_for_host_memory(gpus, current_numa_node);
        RAPIDSMPF_EXPECTS(gpu_count > 0, "No GPUs found in topology", std::runtime_error);
        return get_numa_node_host_memory(current_numa_node)
               / safe_cast<std::uint64_t>(gpu_count);
    }();
    return ret;
}

}  // namespace rapidsmpf
