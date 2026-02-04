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

std::optional<std::string> getenv_optional(std::string_view name) {
    // std::getenv requires a null-terminated string; construct a std::string
    // to ensure this even when called with a non-literal std::string_view.
    char const* value = std::getenv(std::string{name}.c_str());
    if (value == nullptr) {
        return std::nullopt;
    }
    return std::string{value};
}

std::optional<int> getenv_int(std::string_view name) {
    auto value = getenv_optional(name);
    if (!value) {
        return std::nullopt;
    }
    try {
        return std::stoi(*value);
    } catch (...) {
        throw std::runtime_error(
            std::string{"Failed to parse integer from environment variable "}
            + std::string{name} + ": " + *value
        );
    }
}

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
    return getenv_optional("UCX_NET_DEVICES").value_or("");
}

int get_gpu_id() {
    auto cuda_visible = getenv_optional("CUDA_VISIBLE_DEVICES");
    if (cuda_visible) {
        try {
            return std::stoi(*cuda_visible);
        } catch (...) {
            // Ignore parse errors
        }
    }
    return -1;
}

bool is_running_with_rrun() {
    return getenv_optional("RAPIDSMPF_RANK").has_value();
}

bool is_running_with_slurm() {
    return getenv_optional("SLURM_JOB_ID").has_value()
           && getenv_optional("SLURM_PROCID").has_value();
}

bool is_running_with_bootstrap() {
    return is_running_with_rrun();
}

Rank get_rank() {
    if (auto rank_opt = getenv_int("RAPIDSMPF_RANK")) {
        return *rank_opt;
    } else if (auto rank_opt = getenv_int("PMIX_RANK")) {
        return *rank_opt;
    } else if (auto rank_opt = getenv_int("SLURM_PROCID")) {
        return *rank_opt;
    } else {
        return -1;
    }
}

Rank get_nranks() {
    if (!is_running_with_bootstrap()) {
        throw std::runtime_error(
            "get_nranks() can only be called when running with a bootstrap launcher. "
            "Use 'rrun' or 'srun --mpi=pmix' to launch the application."
        );
    } else if (auto nranks_opt = getenv_int("RAPIDSMPF_NRANKS")) {
        return *nranks_opt;
    } else if (auto nranks_opt = getenv_int("SLURM_NPROCS")) {
        return *nranks_opt;
    } else if (auto nranks_opt = getenv_int("SLURM_NTASKS")) {
        return *nranks_opt;
    } else {
        throw std::runtime_error(
            "Could not determine number of ranks. "
            "Ensure RAPIDSMPF_NRANKS, SLURM_NPROCS, or SLURM_NTASKS is set."
        );
    }
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
