/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "rrun_config.hpp"

// NOTE: Do not use RAPIDSMPF_EXPECTS or RAPIDSMPF_FAIL in this file.
// Using these macros introduces a CUDA dependency via rapidsmpf/error.hpp.
// Prefer throwing standard exceptions instead.

namespace rrun {

/** @brief Generate a random 8-character alphanumeric session ID. */
std::string generate_session_id();

/**
 * @brief Detect available GPUs on the system via nvidia-smi.
 * @return Vector of monotonically increasing GPU indices.
 */
std::vector<int> detect_gpus();

/**
 * @brief Check if running under Slurm and populate Slurm-related config fields.
 * @param cfg Configuration to populate with Slurm information.
 * @return true if running under Slurm with required environment variables.
 */
bool detect_slurm_environment(Config& cfg);

/** @brief Parse comma-separated GPU ID string into a vector. */
std::vector<int> parse_gpu_list(std::string const& gpu_str);

/**
 * @brief Fork a child with stdout/stderr redirected to pipes.
 *
 * @param out_fd_stdout File descriptor for reading child's stdout.
 * @param out_fd_stderr File descriptor for reading child's stderr.
 * @param combine_stderr If true, stderr is redirected to stdout pipe.
 * @param child_body Function to execute in the child process (must not return).
 * @return Child PID.
 */
pid_t fork_with_piped_stdio(
    int* out_fd_stdout,
    int* out_fd_stderr,
    bool combine_stderr,
    std::function<void()> child_body
);

/**
 * @brief Execute application via execvp (never returns).
 * @param cfg Configuration containing application binary and arguments.
 */
[[noreturn]] void exec_application(Config const& cfg);

/**
 * @brief Launch a single rank locally (fork-based).
 *
 * @param cfg Configuration.
 * @param global_rank Global rank number (used for RRUN_RANK).
 * @param local_rank Local rank for GPU assignment.
 * @param total_ranks Total number of ranks across all tasks (used for RRUN_NRANKS).
 * @param out_fd_stdout Output file descriptor for stdout.
 * @param out_fd_stderr Output file descriptor for stderr.
 * @return Child process PID.
 */
pid_t launch_rank_local(
    Config const& cfg,
    int global_rank,
    int local_rank,
    int total_ranks,
    int* out_fd_stdout,
    int* out_fd_stderr
);

/**
 * @brief Launch multiple ranks locally using fork.
 *
 * @param cfg Configuration.
 * @param rank_offset Starting global rank for this task.
 * @param ranks_per_task Number of ranks to launch.
 * @param total_ranks Total ranks across all tasks.
 * @return Exit status (0 for success).
 */
int launch_ranks_fork_based(
    Config const& cfg, int rank_offset, int ranks_per_task, int total_ranks
);

/**
 * @brief Set up coordination, launch ranks, and cleanup.
 *
 * Creates coordination infrastructure (socket server or file directory), calls
 * launch_ranks_fork_based, then cleans up.
 *
 * @param cfg Configuration (may modify coord_dir if empty).
 * @param rank_offset Starting global rank for this task.
 * @param ranks_per_task Number of ranks to launch locally.
 * @param total_ranks Total ranks across all tasks.
 * @param coord_dir_hint Hint for coordination directory name (e.g., job ID).
 * @return Exit status (0 for success).
 */
int setup_launch_and_cleanup(
    Config& cfg,
    int rank_offset,
    int ranks_per_task,
    int total_ranks,
    std::string const& coord_dir_hint = ""
);

}  // namespace rrun
