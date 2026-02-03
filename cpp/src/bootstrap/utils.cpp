/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdlib>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sched.h>
#include <unistd.h>

#include <rapidsmpf/bootstrap/utils.hpp>

#if RAPIDSMPF_HAVE_NUMA
#include <numa.h>
#endif

// NOTE: Do not use RAPIDSMPF_EXPECTS or RAPIDSMPF_FAIL in this file.
// Using these macros introduces a CUDA dependency via rapidsmpf/error.hpp.
// Prefer throwing standard exceptions instead.

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

bool is_running_with_slurm() {
    if (std::getenv("SLURM_JOB_ID") != nullptr && std::getenv("SLURM_PROCID") != nullptr)
    {
        return true;
    }
    return false;
}

bool is_running_with_bootstrap() {
    // Only return true if rrun is coordinating (i.e., RAPIDSMPF_RANK is set).
    // Even if Slurm environment variables are present, the user may want to use
    // MPI directly with `srun --mpi=pmix`, so we shouldn't force bootstrap mode
    // unless rrun is explicitly managing the launch.
    return is_running_with_rrun();
}

Rank get_rank() {
    // Check rrun first (explicit configuration takes priority)
    if (char* rank_env = std::getenv("RAPIDSMPF_RANK")) {
        try {
            return std::stoi(rank_env);
        } catch (...) {
            // Ignore parse errors, try next source
        }
    }
    // Check PMIx rank
    if (char* rank_env = std::getenv("PMIX_RANK")) {
        try {
            return std::stoi(rank_env);
        } catch (...) {
            // Ignore parse errors, try next source
        }
    }
    // Check Slurm process ID
    if (char* rank_env = std::getenv("SLURM_PROCID")) {
        try {
            return std::stoi(rank_env);
        } catch (...) {
            // Ignore parse errors
        }
    }
    return -1;
}

Rank get_nranks() {
    if (!is_running_with_bootstrap()) {
        throw std::runtime_error(
            "get_nranks() can only be called when running with a bootstrap launcher. "
            "Use 'rrun' or 'srun --mpi=pmix' to launch the application."
        );
    }

    // Check rrun first (explicit configuration takes priority)
    if (char const* nranks_str = std::getenv("RAPIDSMPF_NRANKS")) {
        try {
            return std::stoi(nranks_str);
        } catch (...) {
            throw std::runtime_error(
                "Failed to parse integer from RAPIDSMPF_NRANKS: "
                + std::string(nranks_str)
            );
        }
    }

    // Check Slurm environment variables
    if (char const* nranks_str = std::getenv("SLURM_NPROCS")) {
        try {
            return std::stoi(nranks_str);
        } catch (...) {
            throw std::runtime_error(
                "Failed to parse integer from SLURM_NPROCS: " + std::string(nranks_str)
            );
        }
    }

    if (char const* nranks_str = std::getenv("SLURM_NTASKS")) {
        try {
            return std::stoi(nranks_str);
        } catch (...) {
            throw std::runtime_error(
                "Failed to parse integer from SLURM_NTASKS: " + std::string(nranks_str)
            );
        }
    }

    throw std::runtime_error(
        "Could not determine number of ranks. "
        "Ensure RAPIDSMPF_NRANKS, SLURM_NPROCS, or SLURM_NTASKS is set."
    );
}

std::vector<int> parse_cpu_list(std::string const& cpulist) {
    std::vector<int> cores;
    if (cpulist.empty()) {
        return cores;
    }

    std::istringstream iss(cpulist);
    std::string token;
    while (std::getline(iss, token, ',')) {
        size_t dash_pos = token.find('-');
        if (dash_pos != std::string::npos) {
            try {
                int start = std::stoi(token.substr(0, dash_pos));
                int end = std::stoi(token.substr(dash_pos + 1));
                for (int i = start; i <= end; ++i) {
                    cores.push_back(i);
                }
            } catch (...) {
                return {};
            }
        } else {
            try {
                cores.push_back(std::stoi(token));
            } catch (...) {
                return {};
            }
        }
    }
    return cores;
}

bool compare_cpu_affinity(std::string const& actual, std::string const& expected) {
    if (actual.empty() && expected.empty()) {
        return true;
    }
    if (actual.empty() || expected.empty()) {
        return false;
    }

    auto actual_cores = parse_cpu_list(actual);
    auto expected_cores = parse_cpu_list(expected);
    std::ranges::sort(actual_cores);
    std::ranges::sort(expected_cores);
    return actual_cores == expected_cores;
}

bool compare_device_lists(std::string const& actual, std::string const& expected) {
    if (actual.empty() && expected.empty()) {
        return true;
    }
    if (actual.empty() || expected.empty()) {
        return false;
    }

    std::vector<std::string> actual_devs;
    std::vector<std::string> expected_devs;

    std::istringstream actual_ss(actual);
    std::string token;
    while (std::getline(actual_ss, token, ',')) {
        actual_devs.push_back(token);
    }

    std::istringstream expected_ss(expected);
    while (std::getline(expected_ss, token, ',')) {
        expected_devs.push_back(token);
    }

    std::ranges::sort(actual_devs);
    std::ranges::sort(expected_devs);
    return actual_devs == expected_devs;
}

}  // namespace rapidsmpf::bootstrap
