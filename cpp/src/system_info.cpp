/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <numa.h>
#include <sched.h>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/system_info.hpp>

namespace rapidsmpf {

int get_current_numa_node_id() noexcept {
    static const int numa_node_id = [] {
        if (!numa_available()) {
            return 0;
        }
        return numa_node_of_cpu(sched_getcpu());
    }();
    return numa_node_id;
}

}  // namespace rapidsmpf
