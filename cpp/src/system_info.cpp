/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <exception>
#include <iostream>

#include <numa.h>
#include <sched.h>
#include <unistd.h>

#include <rapidsmpf/system_info.hpp>

namespace rapidsmpf {

std::uint64_t get_total_host_memory() noexcept {
    static const uint64_t ret = [] {
        auto const page_size = ::sysconf(_SC_PAGE_SIZE);
        auto const phys_pages = ::sysconf(_SC_PHYS_PAGES);

        if (page_size <= 0 || phys_pages <= 0) {
            std::cerr << "get_total_host_memory() - fatal error: "
                      << "sysconf(_SC_PAGE_SIZE/_SC_PHYS_PAGES) failed" << std::endl;
            std::terminate();
        }
        return static_cast<std::uint64_t>(page_size)
               * static_cast<std::uint64_t>(phys_pages);
    }();
    return ret;
}

int get_current_numa_node() noexcept {
    static const int ret = [] {
        if (!numa_available()) {
            return 0;
        }
        return numa_node_of_cpu(sched_getcpu());
    }();
    return ret;
}

std::vector<int> get_current_numa_nodes() noexcept {
    std::vector<int> ret;
    int const cpu = ::sched_getcpu();
    if (numa_available() != -1 && cpu >= 0) {
        int numa_node = numa_node_of_cpu(cpu);
        if (numa_node >= 0) {
            ret.push_back(numa_node);
        }
    }
    if (ret.empty()) {
        return {0};
    }
    return ret;
}

std::uint64_t get_numa_node_host_memory(int numa_id) noexcept {
    if (numa_available() < 0) {
        return get_total_host_memory();
    }
    long long free_ll = 0;  // ignored.
    long long total_ll = numa_node_size64(numa_id, &free_ll);
    if (total_ll < 0) {
        return get_total_host_memory();
    }
    return static_cast<std::uint64_t>(total_ll);
}

}  // namespace rapidsmpf
