/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include <sched.h>
#include <unistd.h>

#include <rapidsmpf/bootstrap/utils.hpp>

#if RAPIDSMPF_HAVE_NUMA
#include <numa.h>
#endif

namespace rapidsmpf::bootstrap {

std::string get_current_cpu_affinity() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    pid_t pid = getpid();
    if (sched_getaffinity(pid, sizeof(cpu_set_t), &cpuset) != 0) {
        return "";
    }

    std::vector<int> cores;
    for (int i = 0; i < static_cast<int>(CPU_SETSIZE); ++i) {
        if (CPU_ISSET(static_cast<unsigned>(i), &cpuset)) {
            cores.push_back(i);
        }
    }

    if (cores.empty()) {
        return "";
    }

    // Format as ranges where possible
    std::ostringstream oss;
    int range_start = cores[0];
    int range_end = cores[0];
    for (size_t i = 1; i < cores.size(); ++i) {
        if (cores[i] == range_end + 1) {
            range_end = cores[i];
        } else {
            if (range_start == range_end) {
                oss << range_start;
            } else {
                oss << range_start << "-" << range_end;
            }
            oss << ",";
            range_start = cores[i];
            range_end = cores[i];
        }
    }
    if (range_start == range_end) {
        oss << range_start;
    } else {
        oss << range_start << "-" << range_end;
    }

    return oss.str();
}

std::vector<int> get_current_numa_nodes() {
    std::vector<int> numa_nodes;
#if RAPIDSMPF_HAVE_NUMA
    if (numa_available() == -1) {
        return numa_nodes;
    }

    // Since processes are typically bound to CPUs on the same NUMA node as their memory,
    // using the CPU's NUMA node (via numa_node_of_cpu) is a reasonable approximation
    // that works well in practice for topology-aware binding scenarios, thus
    // intentionally avoiding the need to get the memory binding policy programmatically
    // for now.
    int cpu = sched_getcpu();
    if (cpu >= 0) {
        int numa_node = numa_node_of_cpu(cpu);
        if (numa_node >= 0) {
            numa_nodes.push_back(numa_node);
        }
    }
#endif
    return numa_nodes;
}

std::string get_ucx_net_devices() {
    char* env = std::getenv("UCX_NET_DEVICES");
    return env ? std::string(env) : std::string();
}

int get_gpu_id() {
    char* cuda_visible = std::getenv("CUDA_VISIBLE_DEVICES");
    if (cuda_visible) {
        try {
            return std::stoi(cuda_visible);
        } catch (...) {
            // Ignore parse errors
        }
    }

    return -1;
}

bool is_running_with_rrun() {
    return std::getenv("RAPIDSMPF_RANK") != nullptr;
}

}  // namespace rapidsmpf::bootstrap
