/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief Process launcher for multi-GPU applications (single-node).
 *
 * rrun is a lightweight alternative to mpirun that:
 * - Launches multiple processes locally without requiring MPI
 * - Automatically assigns GPUs to ranks
 * - Provides file-based coordination for inter-process synchronization
 * - Tags process output with rank numbers (--tag-output feature)
 */

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cucascade/memory/topology_discovery.hpp>

#include <rrun/rrun.hpp>

#include <rapidsmpf/bootstrap/socket_backend.hpp>

#ifdef RAPIDSMPF_HAVE_SLURM
#include <pmix.h>

#include <rapidsmpf/bootstrap/slurm_backend.hpp>
#endif

namespace {

// Forward declarations of mode execution functions (defined later, outside namespace)
struct Config;
[[noreturn]] void execute_slurm_passthrough_mode(Config const& cfg);
int execute_single_node_mode(Config& cfg);

/** A launched child process and its stdout/stderr forwarder threads. */
struct LaunchedProcess {
    pid_t pid;
    std::thread stdout_forwarder;
    std::thread stderr_forwarder;
};
#ifdef RAPIDSMPF_HAVE_SLURM
int execute_slurm_hybrid_mode(Config& cfg);
#endif
int launch_ranks_fork_based(
    Config const& cfg, int rank_offset, int ranks_per_task, int total_ranks
);
[[noreturn]] void exec_application(Config const& cfg);
pid_t launch_rank_local(
    Config const& cfg,
    int global_rank,
    int local_rank,
    int total_ranks,
    int* out_fd_stdout,
    int* out_fd_stderr
);

// NOTE: Do not use RAPIDSMPF_EXPECTS or RAPIDSMPF_FAIL in this file.
// Using these macros introduces a CUDA dependency via rapidsmpf/error.hpp.
// Prefer throwing standard exceptions instead.

static std::mutex output_mutex;

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

/**
 * @brief Generate a random session ID for coordination directory.
 */
std::string generate_session_id() {
    static char const chars[] =
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, sizeof(chars) - 2);

    std::string result;
    result.reserve(8);
    for (int i = 0; i < 8; ++i) {
        result += chars[dis(gen)];
    }
    return result;
}

/**
 * @brief Detect available GPUs on the system.
 *
 * Currently using nvidia-smi to detect GPUs. This may be replaced with NVML in the
 * future.
 *
 * @return Vector of monotonically increasing GPU indices, as observed in nvidia-smi.
 */
std::vector<int> detect_gpus() {
    // Use nvidia-smi to detect GPUs
    FILE* pipe =
        popen("nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null", "r");
    if (!pipe) {
        std::cerr << "[rrun] Warning: Could not detect GPUs using nvidia-smi"
                  << std::endl;
        return {};
    }

    std::vector<int> gpus;
    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        int gpu_id;
        if (sscanf(buffer, "%d", &gpu_id) == 1) {
            gpus.push_back(gpu_id);
        }
    }
    pclose(pipe);
    return gpus;
}

/**
 * @brief Print usage information.
 */
void print_usage(std::string_view prog_name) {
    std::cout
        << "rrun - RapidsMPF Process Launcher\n\n"
        << "Usage: " << prog_name << " [options] <application> [app_args...]\n\n"
        << "Single-Node Options:\n"
        << "  -n <nranks>        Number of ranks to launch (required in single-node "
        << "                     mode)\n"
        << "  -g <gpu_list>      Comma-separated list of GPU IDs (e.g., 0,1,2,3)\n"
        << "                     If not specified, auto-detect available GPUs\n\n"
        << "Slurm Options:\n"
        << "  --slurm            Run in Slurm mode (auto-detected when SLURM_JOB_ID is "
        << "                     set)\n"
        << "                     Two sub-modes:\n"
        << "                     1. Passthrough (no -n): Apply topology bindings and\n"
        << "                     executes application\n"
        << "                     2. Hybrid (with -n): Launch N ranks per Slurm task.\n"
        << "                     In hybrid mode, each Slurm task launches multiple\n"
        << "                     ranks with coordinated global rank numbering.\n"
        << "                     Topology bindings are applied to each rank.\n\n"
        << "Common Options:\n"
        << "  -d <coord_dir>     Coordination directory (default: /tmp/rrun_<random>)\n"
        << "                     Not applicable in Slurm mode\n"
        << "  --tag-output       Tag stdout and stderr with rank number\n"
        << "                     Not applicable in Slurm mode\n"
        << "  --bind-to <type>   Bind to topology resources (default: all)\n"
        << "                     Can be specified multiple times\n"
        << "                     Options: cpu, memory, network, all, none\n"
        << "                     Examples: --bind-to cpu --bind-to network\n"
        << "                               --bind-to none (disable all bindings)\n"
        << "  -x, --set-env <VAR=val>\n"
        << "                     Set environment variable for all ranks\n"
        << "                     Can be specified multiple times\n"
        << "  -v                 Verbose output\n"
        << "  --no-cleanup       Don't cleanup coordination directory on exit\n"
        << "  --file-backend     Use file-based coordination instead of the default\n"
        << "                     socket-based backend. Implies -d if not set.\n"
        << "  -h, --help         Display this help message\n\n"
        << "Environment Variables:\n"
        << "  CUDA_VISIBLE_DEVICES is set for each rank based on GPU assignment\n"
        << "  Additional environment variables can be passed with -x/--set-env\n\n"
        << "Single-Node Examples:\n"
        << "  # Launch 2 ranks with auto-detected GPUs:\n"
        << "  rrun -n 2 ./bench_comm -C ucxx -O all-to-all\n\n"
        << "  # Launch 4 ranks on specific GPUs:\n"
        << "  rrun -n 4 -g 0,1,2,3 ./bench_comm -C ucxx\n\n"
        << "  # Launch with custom environment variables:\n"
        << "  rrun -n 2 -x UCX_TLS=cuda_copy,cuda_ipc,rc,tcp -x MY_VAR=value "
           "./bench_comm\n\n"
        << "Slurm Examples:\n"
        << "  # Passthrough: multiple (4) tasks per node, one task per GPU, two nodes.\n"
        << "  srun --mpi=pmix --nodes=2 --ntasks-per-node=4 --cpus-per-task=36 \\\n"
        << "      --gpus-per-task=1 --gres=gpu:4 \\\n"
        << "      rrun ./benchmarks/bench_shuffle -C ucxx\n\n"
        << "  # Hybrid mode: one task per node, 4 GPUs per task, two nodes.\n"
        << "  srun --mpi=pmix --nodes=2 --ntasks-per-node=1 --cpus-per-task=144 \\\n"
        << "      --gpus-per-task=4 --gres=gpu:4 \\\n"
        << "      rrun -n 4 ./benchmarks/bench_shuffle -C ucxx\n\n"
        << std::endl;
}

/**
 * @brief Check if running under Slurm and populate Slurm-related config fields.
 *
 * @param cfg Configuration to populate with Slurm information.
 * @return true if running under Slurm with required environment variables.
 */
bool detect_slurm_environment(Config& cfg) {
    char const* slurm_job_id = std::getenv("SLURM_JOB_ID");
    char const* slurm_local_id = std::getenv("SLURM_LOCALID");
    char const* slurm_procid = std::getenv("SLURM_PROCID");
    char const* slurm_ntasks = std::getenv("SLURM_NTASKS");

    if (!slurm_job_id || !slurm_local_id) {
        return false;
    }

    try {
        SlurmEnv env;
        env.job_id = std::stoi(slurm_job_id);
        env.local_id = std::stoi(slurm_local_id);
        if (slurm_procid) {
            env.global_rank = std::stoi(slurm_procid);
        }
        if (slurm_ntasks) {
            env.ntasks = std::stoi(slurm_ntasks);
        } else {
            char const* slurm_nprocs = std::getenv("SLURM_NPROCS");
            if (slurm_nprocs) {
                env.ntasks = std::stoi(slurm_nprocs);
            }
        }
        cfg.slurm = std::move(env);
        return true;
    } catch (...) {
        return false;
    }
}

/**
 * @brief Parse GPU list from comma-separated string.
 */
std::vector<int> parse_gpu_list(std::string const& gpu_str) {
    std::vector<int> gpus;
    std::stringstream ss(gpu_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            gpus.push_back(std::stoi(item));
        } catch (...) {
            throw std::runtime_error("Invalid GPU ID: " + item);
        }
    }
    return gpus;
}

/**
 * @brief Parse command-line arguments.
 */
Config parse_args(int argc, char* argv[]) {
    Config cfg;
    int i = 1;
    while (i < argc) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "-n") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing argument for -n");
            }
            cfg.nranks = std::stoi(argv[++i]);
            if (cfg.nranks <= 0) {
                throw std::runtime_error(
                    "Invalid number of ranks: " + std::to_string(cfg.nranks)
                );
            }
        } else if (arg == "-g") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing argument for -g");
            }
            cfg.gpus = parse_gpu_list(argv[++i]);
        } else if (arg == "--tag-output") {
            cfg.tag_output = true;
        } else if (arg == "--bind-to") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing argument for --bind-to");
            }
            std::string bind_type = argv[++i];
            if (bind_type == "none") {
                if (cfg.bind_state == BindToState::Specific
                    || cfg.bind_state == BindToState::All)
                {
                    throw std::runtime_error(
                        "--bind-to none cannot be combined with other --bind-to options"
                    );
                }
                cfg.bind_state = BindToState::None;
                cfg.bind_cpu = false;
                cfg.bind_memory = false;
                cfg.bind_network = false;
            } else if (bind_type == "all") {
                if (cfg.bind_state == BindToState::Specific
                    || cfg.bind_state == BindToState::None)
                {
                    throw std::runtime_error(
                        "--bind-to all cannot be combined with other --bind-to options"
                    );
                }
                cfg.bind_state = BindToState::All;
                cfg.bind_cpu = true;
                cfg.bind_memory = true;
                cfg.bind_network = true;
            } else {
                if (cfg.bind_state == BindToState::None
                    || cfg.bind_state == BindToState::All)
                {
                    throw std::runtime_error(
                        "--bind-to " + bind_type
                        + " cannot be combined with --bind-to none or --bind-to all"
                    );
                }
                cfg.bind_state = BindToState::Specific;
                if (bind_type == "cpu") {
                    cfg.bind_cpu = true;
                } else if (bind_type == "memory") {
                    cfg.bind_memory = true;
                } else if (bind_type == "network") {
                    cfg.bind_network = true;
                } else {
                    throw std::runtime_error(
                        "Invalid --bind-to option: " + bind_type
                        + ". Valid options: cpu, memory, network, all, none"
                    );
                }
            }
        } else if (arg == "-d") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing argument for -d");
            }
            cfg.coord_dir = argv[++i];
        } else if (arg == "-x" || arg == "--set-env") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing argument for -x/--set-env");
            }
            std::string env_spec = argv[++i];
            auto eq_pos = env_spec.find('=');
            if (eq_pos == std::string::npos) {
                throw std::runtime_error(
                    "Invalid environment variable format: " + env_spec
                    + ". Expected VAR=value"
                );
            }
            std::string var_name = env_spec.substr(0, eq_pos);
            std::string var_value = env_spec.substr(eq_pos + 1);
            if (var_name.empty()) {
                throw std::runtime_error("Empty environment variable name");
            }
            cfg.env_vars[var_name] = var_value;
        } else if (arg == "-v") {
            cfg.verbose = true;
        } else if (arg == "--no-cleanup") {
            cfg.cleanup = false;
        } else if (arg == "--file-backend") {
            cfg.file_backend = true;
        } else if (arg == "--slurm") {
            cfg.slurm_mode = true;
        } else if (arg == "--") {
            // Everything after -- is the application and its arguments
            if (i + 1 < argc) {
                cfg.app_binary = argv[i + 1];
                for (int j = i + 2; j < argc; ++j) {
                    cfg.app_args.push_back(argv[j]);
                }
            }
            break;
        } else if (arg[0] == '-') {
            throw std::runtime_error("Unknown option: " + arg);
        } else {
            // First non-option argument is the application binary
            cfg.app_binary = arg;
            // Rest are application arguments
            for (int j = i + 1; j < argc; ++j) {
                cfg.app_args.push_back(argv[j]);
            }
            break;
        }
        ++i;
    }

    // Validate configuration
    if (cfg.app_binary.empty()) {
        throw std::runtime_error("Missing application binary");
    }

    // Auto-detect Slurm mode if not explicitly specified
    if (!cfg.slurm_mode) {
        cfg.slurm_mode = detect_slurm_environment(cfg);
    } else {
        // --slurm was specified, populate Slurm info
        if (!detect_slurm_environment(cfg)) {
            throw std::runtime_error(
                "--slurm specified but required Slurm environment variables "
                "(SLURM_JOB_ID, SLURM_LOCALID) are not set. "
                "Ensure you're running under srun."
            );
        }
    }

    if (cfg.slurm_mode) {
        // Slurm mode validation
        if (!cfg.slurm || cfg.slurm->local_id < 0) {
            throw std::runtime_error(
                "SLURM_LOCALID environment variable not set or invalid"
            );
        }

        // In Slurm mode:
        // - If -n is specified: launch N ranks per Slurm task (hybrid mode)
        // - If -n is not specified: just apply bindings and exec (passthrough mode,
        //                           one rank per task)
        if (cfg.nranks <= 0) {
            cfg.nranks = 1;
        }
    } else {
        // Single-node mode validation
        if (cfg.nranks <= 0) {
            throw std::runtime_error(
                "Number of ranks (-n) must be specified and positive"
            );
        }
    }

    // Auto-detect GPUs if not specified
    if (cfg.gpus.empty()) {
        cfg.gpus = detect_gpus();
        if (cfg.gpus.empty()) {
            std::cerr << "[rrun] Warning: No GPUs detected. CUDA_VISIBLE_DEVICES will "
                         "not be set."
                      << std::endl;
        }
    }

    // Validate GPU count vs rank count (only warn in single-node mode)
    if (!cfg.slurm_mode && !cfg.gpus.empty()
        && cfg.nranks > static_cast<int>(cfg.gpus.size()))
    {
        std::cerr << "[rrun] Warning: Number of ranks (" << cfg.nranks
                  << ") exceeds number of GPUs (" << cfg.gpus.size()
                  << "). Multiple ranks will share GPUs." << std::endl;
    }

    // Generate coordination directory if not specified (not needed in Slurm mode)
    if (cfg.coord_dir.empty() && !cfg.slurm_mode) {
        cfg.coord_dir = "/tmp/rrun_" + generate_session_id();
    }

    // Default to "all" if --bind-to was not explicitly specified
    if (cfg.bind_state == BindToState::NotSpecified) {
        cfg.bind_cpu = true;
        cfg.bind_memory = true;
        cfg.bind_network = true;
    }

    return cfg;
}

/**
 * @brief Helper to fork a child with stdout/stderr redirected to pipes.
 *
 * @param out_fd_stdout The file descriptor for stdout.
 * @param out_fd_stderr The file descriptor for stderr.
 * @param combine_stderr If true, stderr is redirected to stdout pipe.
 * @param child_body The function to execute in the child process. Must not throw, must
 * not return. Must only call exit() or _exit() if an error occurs.
 * @returns Child pid.
 */
pid_t fork_with_piped_stdio(
    int* out_fd_stdout,
    int* out_fd_stderr,
    bool combine_stderr,
    std::function<void()> child_body
) {
    if (out_fd_stdout)
        *out_fd_stdout = -1;
    if (out_fd_stderr)
        *out_fd_stderr = -1;

    int pipe_out[2] = {-1, -1};
    int pipe_err[2] = {-1, -1};
    if (pipe(pipe_out) < 0)
        throw std::runtime_error(
            "Failed to create stdout pipe: " + std::string{std::strerror(errno)}
        );
    if (!combine_stderr) {
        if (pipe(pipe_err) < 0) {
            close(pipe_out[0]);
            close(pipe_out[1]);
            throw std::runtime_error(
                "Failed to create stderr pipe: " + std::string{std::strerror(errno)}
            );
        }
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(pipe_out[0]);
        close(pipe_out[1]);
        if (!combine_stderr) {
            close(pipe_err[0]);
            close(pipe_err[1]);
        }
        throw std::runtime_error("Failed to fork: " + std::string{std::strerror(errno)});
    } else if (pid == 0) {
        // Child: redirect stdout/stderr
        std::ignore = dup2(pipe_out[1], STDOUT_FILENO);
        std::ignore = dup2(combine_stderr ? pipe_out[1] : pipe_err[1], STDERR_FILENO);
        close(pipe_out[0]);
        close(pipe_out[1]);
        if (!combine_stderr) {
            close(pipe_err[0]);
            close(pipe_err[1]);
        }

        // Unbuffered output
        setvbuf(stdout, nullptr, _IONBF, 0);
        setvbuf(stderr, nullptr, _IONBF, 0);

        // Execute child body (should not return on success because
        // exec_application calls execvp).  We must catch any exception here
        // to guarantee the child always reaches _exit(); letting an exception
        // propagate would unwind into the parent code-path where inherited
        // std::thread objects (forwarder threads) trigger std::terminate().
        try {
            child_body();
        } catch (std::exception const& e) {
            fprintf(stderr, "[rrun] Fatal child error: %s\n", e.what());
        } catch (...) {
            fprintf(stderr, "[rrun] Fatal child error (unknown exception)\n");
        }
        _exit(127);
    }

    // Parent: return read fds
    close(pipe_out[1]);
    if (out_fd_stdout)
        *out_fd_stdout = pipe_out[0];
    else
        close(pipe_out[0]);
    if (!combine_stderr) {
        close(pipe_err[1]);
        if (out_fd_stderr)
            *out_fd_stderr = pipe_err[0];
        else
            close(pipe_err[0]);
    }
    return pid;
}

/**
 * @brief Common helper to set up coordination, launch ranks, and cleanup.
 *
 * This function encapsulates the common workflow shared by both Slurm hybrid mode
 * and single-node mode: create coordination directory, launch ranks via fork,
 * cleanup, and report results.
 *
 * A task here denotes a Slurm unit of execution, e.g., a single instance of a
 * program or process, e.g., an instance of the `rrun` executable itself.
 *
 * @param cfg Configuration (will modify coord_dir if empty).
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
) {
    if (cfg.coord_dir.empty()) {
        if (!coord_dir_hint.empty()) {
            cfg.coord_dir = "/tmp/rrun_" + coord_dir_hint;
        } else {
            cfg.coord_dir = "/tmp/rrun_" + generate_session_id();
        }
    }

    // Use an optional to conditionally own the socket server. When --file-backend
    // is set the optional stays empty and the file path is used instead.
    std::optional<rapidsmpf::bootstrap::detail::SocketServer> socket_server;

    if (cfg.file_backend) {
        std::filesystem::create_directories(cfg.coord_dir);
        cfg.env_vars["RRUN_COORD_DIR"] = cfg.coord_dir;
        if (cfg.verbose) {
            std::cout << "[rrun] File coordination directory: " << cfg.coord_dir
                      << std::endl;
        }
    } else {
        // Start the socket coordination server. It binds to 127.0.0.1:0, generates a
        // 256-bit random auth token, and holds all KV/barrier state in memory.
        // Its lifetime covers the entire fork-wait cycle, so child ranks can connect
        // and exchange coordination data without touching the filesystem.
        socket_server.emplace(ranks_per_task);
        cfg.env_vars["RRUN_SOCKET_ADDR"] = socket_server->address();
        cfg.env_vars["RRUN_SOCKET_TOKEN"] = socket_server->token();
        if (cfg.verbose) {
            std::cout << "[rrun] Socket coordination server: " << socket_server->address()
                      << std::endl;
        }
    }

    // Launch ranks and wait for completion
    int exit_status =
        launch_ranks_fork_based(cfg, rank_offset, ranks_per_task, total_ranks);

    if (cfg.file_backend && cfg.cleanup) {
        if (cfg.verbose) {
            std::cout << "[rrun] Cleaning up coordination directory: " << cfg.coord_dir
                      << std::endl;
        }
        std::error_code ec;
        std::filesystem::remove_all(cfg.coord_dir, ec);
        if (ec) {
            std::cerr << "[rrun] Warning: Failed to cleanup directory: " << cfg.coord_dir
                      << ": " << ec.message() << std::endl;
        }
    } else if (cfg.file_backend && cfg.verbose) {
        std::cout << "[rrun] Coordination directory preserved: " << cfg.coord_dir
                  << std::endl;
    }

    if (cfg.verbose && exit_status == 0) {
        std::cout << "\n[rrun] All ranks completed successfully." << std::endl;
    }

    return exit_status;
}

/**
 * @brief Execute application via execvp (never returns).
 *
 * Prepares arguments and calls execvp. On failure, prints error and exits.
 * This function never returns - it either replaces the current process
 * or calls _exit(1) on error.
 *
 * @param cfg Configuration containing application binary and arguments.
 */
[[noreturn]] void exec_application(Config const& cfg) {
    // Prepare arguments for execvp
    std::vector<char*> exec_args;
    exec_args.push_back(const_cast<char*>(cfg.app_binary.c_str()));
    for (auto const& arg : cfg.app_args) {
        exec_args.push_back(const_cast<char*>(arg.c_str()));
    }
    exec_args.push_back(nullptr);

    // Exec the application (this replaces the current process)
    execvp(cfg.app_binary.c_str(), exec_args.data());

    // If we get here, execvp failed
    std::cerr << "[rrun] Failed to execute " << cfg.app_binary << ": "
              << std::strerror(errno) << std::endl;
    _exit(1);
}

#ifdef RAPIDSMPF_HAVE_SLURM

/**
 * @brief Write a value into the FileBackend key-value store.
 *
 * Mirrors the atomic write-then-rename pattern used by FileBackend::write_file
 * so that the children's `get()` call never sees partial content.
 *
 * @param coord_dir Local coordination directory (RRUN_COORD_DIR).
 * @param key       Key name (e.g. "ucxx_root_address").
 * @param value     Binary-safe value to store.
 */
void write_kv_for_children(
    std::string const& coord_dir, std::string const& key, std::string_view value
) {
    std::string kv_dir = coord_dir + "/kv";
    std::filesystem::create_directories(kv_dir);

    std::string path = kv_dir + "/" + key;
    std::string tmp_path = path + ".tmp." + std::to_string(getpid());

    std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error("Failed to open temporary file: " + tmp_path);
    }
    ofs.write(value.data(), static_cast<std::streamsize>(value.size()));
    ofs.close();

    std::error_code ec;
    std::filesystem::rename(tmp_path, path, ec);
    if (ec) {
        std::error_code rm_ec;
        std::filesystem::remove(tmp_path, rm_ec);
        throw std::runtime_error(
            "Failed to rename " + tmp_path + " to " + path + ": " + ec.message()
        );
    }
}

/**
 * @brief Relay the root UCXX address from rank 0 to all Slurm tasks.
 *
 * Called in a background thread after all ranks have been launched.  The root
 * parent (SLURM_PROCID==0) polls the address file that rank 0 writes when it
 * initialises UCXX.  Once available it publishes the address via PMIx so that
 * non-root parents can retrieve it.  Every parent then writes the address into
 * its local FileBackend kv store so that the children can pick it up with
 * `get("ucxx_root_address")`.
 *
 * If rank 0 never writes the address (e.g. diagnostic tools that don't use
 * UCXX) the thread simply waits until @p stop becomes true (set by the caller
 * after all children have exited) and returns without doing anything.
 *
 * @param cfg           Configuration.
 * @param address_file  Path to the file rank 0 writes its address to.
 * @param stop          Atomic flag set by the main thread once children exit.
 */
void relay_root_address(
    Config const& cfg, std::string const& address_file, std::atomic<bool> const& stop
) {
    bool is_root_parent = (cfg.slurm->global_rank == 0);
    std::string encoded_address;

    if (is_root_parent) {
        // Poll for the address file from rank 0.
        while (!stop.load(std::memory_order_relaxed)) {
            if (std::filesystem::exists(address_file)) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        if (stop.load(std::memory_order_relaxed)
            && !std::filesystem::exists(address_file))
        {
            // Children exited without rank 0 writing the address; nothing to relay.
            return;
        }

        std::ifstream addr_stream(address_file);
        if (!addr_stream) {
            std::cerr << "[rrun] Warning: failed to open root address file: "
                      << address_file << std::endl;
            return;
        }
        std::getline(addr_stream, encoded_address);
        addr_stream.close();
        std::filesystem::remove(address_file);

        if (encoded_address.empty()) {
            std::cerr << "[rrun] Warning: root address file was empty" << std::endl;
            return;
        }

        if (cfg.verbose) {
            std::cout << "[rrun] Got root address from rank 0 (hex-encoded, "
                      << encoded_address.size() << " chars)" << std::endl;
        }
    }

    // PMIx coordination between parents.
    int parent_rank = cfg.slurm->global_rank;
    int parent_nranks = cfg.slurm->ntasks;

    rapidsmpf::bootstrap::Context parent_ctx{
        parent_rank,
        parent_nranks,
        rapidsmpf::bootstrap::BackendType::SLURM,
        std::nullopt,
        nullptr
    };

    auto backend =
        std::make_shared<rapidsmpf::bootstrap::detail::SlurmBackend>(parent_ctx);

    if (is_root_parent) {
        if (cfg.verbose) {
            std::cout << "[rrun] Publishing root address via PMIx (hex-encoded, "
                      << encoded_address.size() << " chars)" << std::endl;
        }
        backend->put("rapidsmpf_root_address", encoded_address);
    }

    backend->sync();

    if (!is_root_parent) {
        encoded_address =
            backend->get("rapidsmpf_root_address", std::chrono::seconds{30});
        if (cfg.verbose) {
            std::cout << "[rrun] Retrieved root address via PMIx (hex-encoded, "
                      << encoded_address.size() << " chars)" << std::endl;
        }
    }

    // Write into the local FileBackend kv store so children can get() it.
    // The raw address from UCXX is hex-encoded, but the FileBackend stores
    // opaque bytes.  The UCXX bootstrap Path 3 stores the *raw* address via
    // put(), so we must hex-decode before writing here.
    //
    // However, the UCXX bootstrap Path 4 retrieves the value as a raw string
    // and passes it directly to createAddressFromString.  Path 3's put() writes
    // the raw string_view from the UCXX address.  So the FileBackend kv entry
    // must contain the *raw* (binary) address, not the hex-encoded one.
    //
    // The hex encoding is only used for the address file and PMIx transport
    // (which may not be binary-safe).  Decode before writing to the kv store.

    // Inline hex decode (same as in ucxx.cpp).
    auto hex_decode = [](std::string_view const& input) -> std::string {
        std::string result;
        result.reserve(input.size() / 2);
        for (std::size_t i = 0; i + 1 < input.size(); i += 2) {
            auto high = static_cast<unsigned char>(
                (input[i] >= 'a') ? (input[i] - 'a' + 10) : (input[i] - '0')
            );
            auto low = static_cast<unsigned char>(
                (input[i + 1] >= 'a') ? (input[i + 1] - 'a' + 10) : (input[i + 1] - '0')
            );
            result.push_back(static_cast<char>((high << 4) | low));
        }
        return result;
    };

    std::string raw_address = hex_decode(encoded_address);
    write_kv_for_children(cfg.coord_dir, "ucxx_root_address", raw_address);

    if (cfg.verbose) {
        std::cout << "[rrun] Wrote root address to local kv store for children"
                  << std::endl;
    }
}

/**
 * @brief Execute application in Slurm hybrid mode with PMIx coordination.
 *
 * All ranks are launched simultaneously.  A background thread relays the root
 * UCXX address (if any) between the parents via PMIx and writes it into each
 * parent's local FileBackend kv store so that the children can retrieve it
 * through the normal bootstrap Path 3/4 (put/get).
 *
 * Applications that never initialise UCXX communication work transparently:
 * the relay thread detects that all children have exited without producing an
 * address and returns without action.
 *
 * @param cfg Configuration.
 * @return Exit status (0 for success).
 */
int execute_slurm_hybrid_mode(Config& cfg) {
    if (!cfg.slurm || cfg.slurm->job_id < 0 || cfg.slurm->ntasks <= 0
        || cfg.slurm->global_rank < 0)
    {
        throw std::runtime_error(
            "SLURM_JOB_ID, SLURM_NTASKS and SLURM_PROCID must be set for Slurm hybrid "
            "mode. Ensure you are running under srun with --ntasks (or equivalent)."
        );
    }

    int total_ranks = cfg.slurm->ntasks * cfg.nranks;
    int rank_offset = cfg.slurm->global_rank * cfg.nranks;
    std::string coord_hint = "slurm_" + std::to_string(cfg.slurm->job_id);

    if (cfg.verbose) {
        std::cout << "[rrun] Slurm hybrid mode: task " << cfg.slurm->global_rank
                  << " launching " << cfg.nranks << " ranks per task" << std::endl;
    }

    // Slurm hybrid mode uses the FileBackend for intra-task coordination.
    // The SocketServer is currently task-local only.
    cfg.file_backend = true;

    // Set up coordination directory (created by setup_launch_and_cleanup).
    if (cfg.coord_dir.empty()) {
        cfg.coord_dir = "/tmp/rrun_slurm_" + std::to_string(cfg.slurm->job_id);
    }

    // Set RRUN_ROOT_ADDRESS_FILE for rank 0 (only on the root parent).
    // Rank 0's UCXX bootstrap (Path 1) writes its listener address here.
    std::string address_file =
        "/tmp/rapidsmpf_root_address_" + std::to_string(cfg.slurm->job_id);
    bool is_root_parent = (cfg.slurm->global_rank == 0);
    if (is_root_parent) {
        setenv("RRUN_ROOT_ADDRESS_FILE", address_file.c_str(), 1);
    }

    // Launch all ranks (including rank 0) simultaneously.
    // The ranks use FileBackend (Path 3/4) for UCXX bootstrap coordination.
    // The relay thread below will populate the local kv store with the root
    // address once it becomes available.
    std::atomic<bool> relay_stop{false};
    std::thread relay_thread(
        relay_root_address, std::cref(cfg), std::cref(address_file), std::cref(relay_stop)
    );

    int exit_status =
        setup_launch_and_cleanup(cfg, rank_offset, cfg.nranks, total_ranks, coord_hint);

    // Children have exited; tell the relay thread to stop if it hasn't already.
    relay_stop.store(true, std::memory_order_relaxed);

    unsetenv("RRUN_ROOT_ADDRESS_FILE");

    if (relay_thread.joinable()) {
        relay_thread.join();
    }

    return exit_status;
}
#endif  // RAPIDSMPF_HAVE_SLURM

/**
 * @brief Execute application in single-node mode with FILE backend.
 *
 * Uses fork-based execution with file-based coordination.
 *
 * @param cfg Configuration.
 * @return Exit status (0 for success).
 */
int execute_single_node_mode(Config& cfg) {
    if (cfg.verbose) {
        std::cout << "[rrun] Single-node mode: launching " << cfg.nranks << " ranks"
                  << std::endl;
    }

    return setup_launch_and_cleanup(cfg, 0, cfg.nranks, cfg.nranks);
}

/**
 * @brief Execute application in Slurm passthrough mode (single rank per task).
 *
 * Applies topology bindings and executes the application directly without forking.
 * This function never returns - it either replaces the current process or exits on error.
 *
 * @param cfg Configuration.
 */
[[noreturn]] void execute_slurm_passthrough_mode(Config const& cfg) {
    if (!cfg.slurm || cfg.slurm->ntasks <= 0) {
        throw std::runtime_error(
            "SLURM_NTASKS must be set for Slurm passthrough mode. "
            "Ensure you are running under srun with --ntasks (or equivalent)."
        );
    }

    if (cfg.verbose) {
        std::cout << "[rrun] Slurm passthrough mode: applying bindings and exec'ing"
                  << std::endl;
    }

    // Discover topology BEFORE narrowing CUDA_VISIBLE_DEVICES.
    cucascade::memory::topology_discovery discovery;
    bool const have_topology = discovery.discover();

    // Set rrun coordination environment variables so the application knows
    // it's being launched by rrun and should use bootstrap mode
    setenv("RRUN_RANK", std::to_string(cfg.slurm->global_rank).c_str(), 1);
    setenv("RRUN_NRANKS", std::to_string(cfg.slurm->ntasks).c_str(), 1);

    // Determine GPU for this Slurm task
    int gpu_id = -1;
    if (!cfg.gpus.empty()) {
        gpu_id =
            cfg.gpus[static_cast<std::size_t>(cfg.slurm->local_id) % cfg.gpus.size()];
        setenv("CUDA_VISIBLE_DEVICES", std::to_string(gpu_id).c_str(), 1);

        if (cfg.verbose) {
            std::cout << "[rrun] Slurm task (passthrough) local_id="
                      << cfg.slurm->local_id << " assigned to GPU " << gpu_id
                      << std::endl;
        }
    }

    // Set custom environment variables
    for (auto const& env_pair : cfg.env_vars) {
        setenv(env_pair.first.c_str(), env_pair.second.c_str(), 1);
    }

    if (have_topology) {
        try {
            rapidsmpf::rrun::bind(
                discovery.get_topology(),
                gpu_id >= 0
                    ? std::optional<unsigned int>(static_cast<unsigned int>(gpu_id))
                    : std::nullopt,
                {.cpu = cfg.bind_cpu,
                 .memory = cfg.bind_memory,
                 .network = cfg.bind_network}
            );
        } catch (std::exception const& e) {
            std::cerr << "[rrun] Warning: " << e.what() << std::endl;
        }
    } else if (cfg.verbose) {
        std::cerr << "[rrun] Warning: topology discovery failed; "
                  << "resource binding skipped." << std::endl;
    }

    exec_application(cfg);
}

/**
 * @brief Launch multiple ranks locally using fork.
 *
 * A task here denotes a Slurm unit of execution, e.g., a single instance of a
 * program or process, e.g., an instance of the `rrun` executable itself.
 *
 * @param cfg Configuration.
 * @param rank_offset Starting global rank for this task.
 * @param ranks_per_task Number of ranks to launch.
 * @param total_ranks Total ranks across all tasks.
 * @return Exit status (0 for success).
 */
int launch_ranks_fork_based(
    Config const& cfg, int rank_offset, int ranks_per_task, int total_ranks
) {
    std::vector<LaunchedProcess> processes;
    processes.reserve(static_cast<std::size_t>(ranks_per_task));

    // Block SIGINT/SIGTERM in this thread; a dedicated thread will handle them.
    sigset_t signal_set;
    sigemptyset(&signal_set);
    sigaddset(&signal_set, SIGINT);
    sigaddset(&signal_set, SIGTERM);
    sigprocmask(SIG_BLOCK, &signal_set, nullptr);

    auto suppress_output = std::make_shared<std::atomic<bool>>(false);

    // Helper to create a forwarder thread for a given fd (returns default thread if fd <
    // 0).
    auto make_forwarder = [&](int fd, int rank, bool to_stderr) -> std::thread {
        if (fd < 0)
            return {};
        return std::thread([fd, rank, to_stderr, &cfg, suppress_output]() {
            FILE* stream = fdopen(fd, "r");
            if (!stream) {
                close(fd);
                return;
            }
            std::string tag =
                cfg.tag_output ? ("[" + std::to_string(rank) + "] ") : std::string{};
            char buffer[4096];
            while (fgets(buffer, sizeof(buffer), stream) != nullptr) {
                if (suppress_output->load(std::memory_order_relaxed)) {
                    continue;
                }
                FILE* out = to_stderr ? stderr : stdout;
                {
                    std::lock_guard<std::mutex> lock(output_mutex);
                    if (!tag.empty()) {
                        fputs(tag.c_str(), out);
                    }
                    fputs(buffer, out);
                    fflush(out);
                }
            }
            fclose(stream);
        });
    };

    for (int local_rank = 0; local_rank < ranks_per_task; ++local_rank) {
        int global_rank = rank_offset + local_rank;
        int fd_out = -1;
        int fd_err = -1;
        pid_t pid = launch_rank_local(
            cfg, global_rank, local_rank, total_ranks, &fd_out, &fd_err
        );

        if (cfg.verbose) {
            std::ostringstream msg;
            msg << "[rrun] Launched rank " << global_rank << " (PID " << pid << ")";
            if (!cfg.gpus.empty()) {
                msg << " on GPU "
                    << cfg.gpus[static_cast<std::size_t>(local_rank) % cfg.gpus.size()];
            }
            msg << std::endl;
            std::string msg_str = msg.str();

            std::cout << msg_str;
            std::cout.flush();
        }
        processes.emplace_back(
            pid,
            make_forwarder(fd_out, global_rank, false),
            make_forwarder(fd_err, global_rank, true)
        );
    }

    // Forward signals to all processes (including pre-launched).
    std::thread([signal_set, &processes, suppress_output]() mutable {
        for (;;) {
            int sig = 0;
            int rc = sigwait(&signal_set, &sig);
            if (rc != 0) {
                continue;
            }
            suppress_output->store(true, std::memory_order_relaxed);
            for (auto& p : processes) {
                kill(p.pid, sig);
            }
            return;
        }
    }).detach();

    std::cout << "\n[rrun] All ranks launched. Waiting for completion...\n" << std::endl;

    // Wait for all processes
    int exit_status = 0;
    for (std::size_t i = 0; i < processes.size(); ++i) {
        LaunchedProcess& proc = processes[i];
        int status = 0;
        int global_rank = rank_offset + static_cast<int>(i);
        if (waitpid(proc.pid, &status, 0) < 0) {
            std::cerr << "[rrun] Failed to wait for rank " << global_rank << " (PID "
                      << proc.pid << "): " << std::strerror(errno) << std::endl;
            exit_status = 1;
            continue;
        }

        if (WIFEXITED(status)) {
            int code = WEXITSTATUS(status);
            if (code != 0) {
                std::cerr << "[rrun] Rank " << global_rank << " (PID " << proc.pid
                          << ") exited with code " << code << std::endl;
                exit_status = code;
            }
        } else if (WIFSIGNALED(status)) {
            int sig = WTERMSIG(status);
            std::cerr << "[rrun] Rank " << global_rank << " (PID " << proc.pid
                      << ") terminated by signal " << sig << std::endl;
            exit_status = 128 + sig;
        }
    }

    // Wait for forwarder threads to finish
    for (auto& p : processes) {
        if (p.stdout_forwarder.joinable()) {
            p.stdout_forwarder.join();
        }
        if (p.stderr_forwarder.joinable()) {
            p.stderr_forwarder.join();
        }
    }

    return exit_status;
}

/**
 * @brief Launch a single rank locally (fork-based).
 *
 * @param cfg Configuration.
 * @param global_rank Global rank number (used for RRUN_RANK).
 * @param local_rank Local rank for GPU assignment (defaults to global_rank).
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
) {
    return fork_with_piped_stdio(
        out_fd_stdout,
        out_fd_stderr,
        /*combine_stderr*/ false,
        [&cfg, global_rank, local_rank, total_ranks]() {
            // Discover topology BEFORE narrowing CUDA_VISIBLE_DEVICES to a
            // single GPU, otherwise the discovery layer only sees the one
            // device and GPU-ID lookup by physical ID will fail.
            cucascade::memory::topology_discovery discovery;
            bool const have_topology = discovery.discover();

            // Set custom environment variables first (can be overridden by specific vars)
            for (auto const& env_pair : cfg.env_vars) {
                setenv(env_pair.first.c_str(), env_pair.second.c_str(), 1);
            }

            setenv("RRUN_RANK", std::to_string(global_rank).c_str(), 1);
            setenv("RRUN_NRANKS", std::to_string(total_ranks).c_str(), 1);

            // In Slurm hybrid mode, unset Slurm/PMIx rank variables to avoid confusion
            // Children should not try to initialize PMIx themselves
            if (cfg.slurm_mode) {
                unsetenv("SLURM_PROCID");
                unsetenv("SLURM_LOCALID");
                unsetenv("PMIX_RANK");
                unsetenv("PMIX_NAMESPACE");
            }

            // Set CUDA_VISIBLE_DEVICES if GPUs are available
            // Use local_rank for GPU assignment (for Slurm hybrid mode)
            int gpu_id = -1;
            if (!cfg.gpus.empty()) {
                gpu_id = cfg.gpus[static_cast<std::size_t>(local_rank) % cfg.gpus.size()];
                setenv("CUDA_VISIBLE_DEVICES", std::to_string(gpu_id).c_str(), 1);
            }

            if (have_topology) {
                try {
                    rapidsmpf::rrun::bind(
                        discovery.get_topology(),
                        gpu_id >= 0 ? std::optional<unsigned int>(
                                          static_cast<unsigned int>(gpu_id)
                                      )
                                    : std::nullopt,
                        {.cpu = cfg.bind_cpu,
                         .memory = cfg.bind_memory,
                         .network = cfg.bind_network}
                    );
                } catch (std::exception const& e) {
                    fprintf(stderr, "[rrun] Warning: %s\n", e.what());
                }
            } else if (cfg.verbose) {
                fprintf(
                    stderr,
                    "[rrun] Warning: topology discovery failed for rank %d; "
                    "resource binding skipped.\n",
                    global_rank
                );
            }

            exec_application(cfg);
        }
    );
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        // Parse arguments
        Config cfg = parse_args(argc, argv);

        if (cfg.verbose) {
            std::cout << "rrun configuration:\n";
            std::cout << "  Mode:          " << (cfg.slurm_mode ? "Slurm" : "Single-node")
                      << "\n"
                      << "  GPUs:          ";
            if (cfg.gpus.empty()) {
                std::cout << "(none)\n";
            } else {
                for (std::size_t i = 0; i < cfg.gpus.size(); ++i) {
                    if (i > 0)
                        std::cout << ", ";
                    std::cout << cfg.gpus[i];
                }
                std::cout << "\n";
            }
            if (cfg.slurm_mode) {
                std::cout << "  Slurm Local ID: " << cfg.slurm->local_id << "\n"
                          << "  Slurm Rank:     " << cfg.slurm->global_rank << "\n"
                          << "  Slurm NTasks:   " << cfg.slurm->ntasks << "\n";
            } else {
                if (cfg.tag_output) {
                    std::cout << "  Tag Output:    Yes\n";
                }
                std::cout << "  Ranks:         " << cfg.nranks << "\n"
                          << "  Backend:       " << (cfg.file_backend ? "file" : "socket")
                          << "\n";
                if (cfg.file_backend) {
                    std::cout << "  Coord Dir:     " << cfg.coord_dir << "\n"
                              << "  Cleanup:       " << (cfg.cleanup ? "yes" : "no")
                              << "\n";
                }
            }
            std::cout << "  Application:   " << cfg.app_binary << "\n";
            std::vector<std::string> bind_types;
            if (cfg.bind_cpu)
                bind_types.push_back("cpu");
            if (cfg.bind_memory)
                bind_types.push_back("memory");
            if (cfg.bind_network)
                bind_types.push_back("network");
            if (bind_types.empty()) {
                std::cout << "  Bind To:       none\n";
            } else {
                std::cout << "  Bind To:       ";
                for (std::size_t i = 0; i < bind_types.size(); ++i) {
                    if (i > 0)
                        std::cout << ", ";
                    std::cout << bind_types[i];
                }
                std::cout << "\n";
            }
            if (!cfg.env_vars.empty()) {
                std::cout << "  Env Vars:      ";
                bool first = true;
                for (auto const& env_pair : cfg.env_vars) {
                    if (!first)
                        std::cout << "                 ";
                    std::cout << env_pair.first << "=" << env_pair.second << "\n";
                    first = false;
                }
            }
            std::cout << std::endl;
        }

        if (cfg.slurm_mode) {
            if (cfg.nranks == 1) {
                // Slurm passthrough mode: single rank per task, no forking
                execute_slurm_passthrough_mode(cfg);
            }
            // Slurm hybrid mode: multiple ranks per task with PMIx coordination
#ifdef RAPIDSMPF_HAVE_SLURM
            return execute_slurm_hybrid_mode(cfg);
#else
            std::cerr << "[rrun] Error: Slurm hybrid mode requires PMIx support but "
                      << "rapidsmpf was not built with PMIx." << std::endl;
            std::cerr << "[rrun] Rebuild with -DBUILD_SLURM_SUPPORT=ON or use "
                         "passthrough mode "
                      << "(without -n flag)." << std::endl;
            return 1;
#endif
        } else {
            // Single-node mode with file backend
            return execute_single_node_mode(cfg);
        }

    } catch (std::exception const& e) {
        std::cerr << "[rrun] Error: " << e.what() << std::endl;
        std::cerr << "[rrun] Run with -h or --help for usage information." << std::endl;
        return 1;
    }
}
