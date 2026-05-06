/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

// NOTE: Do not use RAPIDSMPF_EXPECTS or RAPIDSMPF_FAIL in this file.
// Using these macros introduces a CUDA dependency via rapidsmpf/error.hpp.
// Prefer throwing standard exceptions instead.

namespace rrun {

/**
 * @brief State of --bind-to option specification.
 */
enum class BindToState {
    NotSpecified,  // Default, will be treated as "all"
    None,  // --bind-to none
    All,  // --bind-to all
    Specific  // --bind-to cpu/memory/network (one or more)
};

/**
 * @brief Slurm environment info (from SLURM_* variables).
 * Only present when running under Slurm. Required fields must be set before
 * using hybrid or passthrough mode; missing values cause runtime_error.
 */
struct SlurmEnv {
    int job_id{-1};  // SLURM_JOB_ID
    int local_id{-1};  // SLURM_LOCALID
    int global_rank{-1};  // SLURM_PROCID
    int ntasks{-1};  // SLURM_NTASKS or SLURM_NPROCS
};

/**
 * @brief Configuration for the rrun launcher.
 */
struct Config {
    int nranks{1};  // Total number of ranks
    std::string app_binary;  // Application binary path
    std::vector<std::string> app_args;  // Arguments to pass to application
    std::vector<int> gpus;  // GPU IDs to use
    std::string coord_dir;  // Coordination directory
    std::map<std::string, std::string> env_vars;  // Environment variables to pass
    bool verbose{false};  // Verbose output
    bool cleanup{true};  // Cleanup coordination directory on exit
    bool tag_output{false};  // Tag output with rank number
    bool bind_cpu{false};  // Bind to CPU affinity
    bool bind_memory{false};  // Bind to NUMA memory
    bool bind_network{false};  // Bind to network devices
    BindToState bind_state{
        BindToState::NotSpecified
    };  // State of --bind-to specification
    bool file_backend{false};  // Force file-based coordination instead of socket
    bool slurm_mode{false};  // Running under Slurm (--slurm or auto-detected)
    std::optional<SlurmEnv> slurm;  // Set when slurm_mode is true
};

/** A launched child process and its stdout/stderr forwarder threads. */
struct LaunchedProcess {
    pid_t pid;
    std::thread stdout_forwarder;
    std::thread stderr_forwarder;
};

/** Mutex protecting interleaved stdout/stderr writes from forwarder threads. */
extern std::mutex output_mutex;

}  // namespace rrun
