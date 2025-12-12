/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <numa.h>
#include <sched.h>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/system_info.hpp>

namespace rapidsmpf {

std::uint64_t get_total_host_memory() noexcept {
    static const uint64_t ret = [] {
        auto const page_size = ::sysconf(_SC_PAGE_SIZE);
        auto const phys_pages = ::sysconf(_SC_PHYS_PAGES);

        // Because `get_total_host_memory()` is marked `noexcept`, any failure
        // here results in process termination, which is intentional.
        RAPIDSMPF_EXPECTS(
            page_size > 0 && phys_pages > 0,
            "sysconf(_SC_PAGE_SIZE/_SC_PHYS_PAGES) failed",
            std::runtime_error
        );
        return static_cast<std::uint64_t>(page_size)
               * static_cast<std::uint64_t>(phys_pages);
    }();
    return ret;
}

int get_current_numa_node_id() noexcept {
    static const int ret = [] {
        if (!numa_available()) {
            return 0;
        }
        return numa_node_of_cpu(sched_getcpu());
    }();
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
