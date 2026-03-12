/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <sched.h>
#include <unistd.h>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/system_info.hpp>


#if RAPIDSMPF_HAVE_NUMA
#include <numa.h>
#endif

namespace rapidsmpf {

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

std::vector<int> get_current_numa_nodes() noexcept {
    std::vector<int> ret;
#if RAPIDSMPF_HAVE_NUMA
    int const cpu = ::sched_getcpu();
    if (numa_available() != -1 && cpu >= 0) {
        int numa_node = numa_node_of_cpu(cpu);
        if (numa_node >= 0) {
            ret.push_back(numa_node);
        }
    }
#endif
    if (ret.empty()) {
        return {0};
    }
    return ret;
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

}  // namespace rapidsmpf
