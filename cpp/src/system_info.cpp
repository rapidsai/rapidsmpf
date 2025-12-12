/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */


#if RAPIDSMPF_HAVE_NUMA
#include <numa.h>
#include <sched.h>
#endif

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/system_info.hpp>

namespace rapidsmpf {

int get_current_numa_node_id() {
#if RAPIDSMPF_HAVE_NUMA
    static const int numa_node_id = [] {
        RAPIDSMPF_EXPECTS(
            numa_available() != -1, "NUMA is not available", std::runtime_error
        );
        int cpu = sched_getcpu();
        int numa_node = numa_node_of_cpu(cpu);
        RAPIDSMPF_EXPECTS(
            numa_node >= 0, "failed to get NUMA node ID", std::runtime_error
        );
        return numa_node;
    }();
    return numa_node_id;
#else
    return 0;
#endif
}

}  // namespace rapidsmpf
