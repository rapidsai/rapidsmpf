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
#include <chrono>
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

#include <sched.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#if RAPIDSMPF_HAVE_NUMA
#include <numa.h>
#endif

#include <cucascade/memory/topology_discovery.hpp>

// NOTE: Do not use RAPIDSMPF_EXPECTS or RAPIDSMPF_FAIL in this file.
// Using these macros introduces a CUDA dependency via rapidsmpf/error.hpp.
// Prefer throwing standard exceptions instead.

namespace {

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
    std::optional<cucascade::memory::system_topology_info>
        topology;  // Discovered topology information
    std::map<int, cucascade::memory::gpu_topology_info const*>
        gpu_topology_map;  // Map GPU ID to topology info
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
        std::cerr << "Warning: Could not detect GPUs using nvidia-smi" << std::endl;
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
        << "  -n <nranks>        Number of ranks to launch (required)\n"
        << "  -g <gpu_list>      Comma-separated list of GPU IDs (e.g., 0,1,2,3)\n"
        << "                     If not specified, auto-detect available GPUs\n\n"
        << "Common Options:\n"
        << "  -d <coord_dir>     Coordination directory (default: /tmp/rrun_<random>)\n"
        << "  --tag-output       Tag stdout and stderr with rank number\n"
        << "  --bind-to <type>   Bind to topology resources (default: all)\n"
        << "                     Can be specified multiple times\n"
        << "                     Options: cpu, memory, network, all, none\n"
        << "                     Examples: --bind-to cpu --bind-to network\n"
        << "                              --bind-to none (disable all bindings)\n"
        << "  -x, --set-env <VAR=val>\n"
        << "                     Set environment variable for all ranks\n"
        << "                     Can be specified multiple times\n"
        << "  -v                 Verbose output\n"
        << "  --no-cleanup       Don't cleanup coordination directory on exit\n"
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
        << std::endl;
}

/**
 * @brief Parse CPU list string into CPU mask for sched_setaffinity.
 *
 * Accepts formats like "0-31,128-159" or comma-separated single cores.
 * Returns a cpu_set_t mask that can be used with sched_setaffinity.
 *
 * @param cpulist CPU list string (e.g., "0-31,128-159").
 * @param cpuset Output CPU set to populate.
 * @return true on success, false on failure.
 */
bool parse_cpu_list_to_mask(std::string const& cpulist, cpu_set_t* cpuset) {
    CPU_ZERO(cpuset);
    if (cpulist.empty()) {
        return false;
    }

    std::istringstream iss(cpulist);
    std::string token;
    while (std::getline(iss, token, ',')) {
        size_t dash_pos = token.find('-');
        if (dash_pos != std::string::npos) {
            // Range, e.g., "0-31"
            try {
                int start = std::stoi(token.substr(0, dash_pos));
                int end = std::stoi(token.substr(dash_pos + 1));
                for (int i = start; i <= end; ++i) {
                    if (i >= 0 && i < static_cast<int>(CPU_SETSIZE)) {
                        CPU_SET(static_cast<unsigned>(i), cpuset);
                    }
                }
            } catch (...) {
                return false;
            }
        } else {
            // Single core, e.g., "5"
            try {
                int core = std::stoi(token);
                if (core >= 0 && core < static_cast<int>(CPU_SETSIZE)) {
                    CPU_SET(static_cast<unsigned>(core), cpuset);
                }
            } catch (...) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Set CPU affinity for the current process.
 *
 * @param cpu_affinity_list CPU affinity list string (e.g., "0-31,128-159"), as in the
 * format of `cucascade::memory::gpu_topology_info::cpu_affinity_list`.
 * @return true on success, false on failure.
 */
bool set_cpu_affinity(std::string const& cpu_affinity_list) {
    if (cpu_affinity_list.empty()) {
        return false;
    }

    cpu_set_t cpuset;
    if (!parse_cpu_list_to_mask(cpu_affinity_list, &cpuset)) {
        return false;
    }

    pid_t pid = getpid();
    if (sched_setaffinity(pid, sizeof(cpu_set_t), &cpuset) != 0) {
        return false;
    }

    return true;
}

/**
 * @brief Set NUMA memory binding for the current process.
 *
 * @param memory_binding Vector of NUMA node IDs to bind memory to.
 * @return true on success, false on failure or if NUMA is not available.
 */
bool set_numa_memory_binding(std::vector<int> const& memory_binding) {
#if RAPIDSMPF_HAVE_NUMA
    if (memory_binding.empty()) {
        return false;
    }

    if (numa_available() == -1) {
        return false;
    }

    struct bitmask* nodemask = numa_allocate_nodemask();
    if (!nodemask) {
        return false;
    }

    numa_bitmask_clearall(nodemask);
    for (int node : memory_binding) {
        if (node >= 0) {
            numa_bitmask_setbit(nodemask, static_cast<unsigned int>(node));
        }
    }

    numa_set_membind(nodemask);
    numa_free_nodemask(nodemask);

    return true;
#else
    std::ignore = memory_binding;  // Suppress unused parameter warning
    return false;
#endif
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

    // Single-node mode validation
    if (cfg.nranks <= 0) {
        throw std::runtime_error("Number of ranks (-n) must be specified and positive");
    }

    // Auto-detect GPUs if not specified
    if (cfg.gpus.empty()) {
        cfg.gpus = detect_gpus();
        if (cfg.gpus.empty()) {
            std::cerr
                << "Warning: No GPUs detected. CUDA_VISIBLE_DEVICES will not be set."
                << std::endl;
        }
    }

    // Validate GPU count vs rank count
    if (!cfg.gpus.empty() && cfg.nranks > static_cast<int>(cfg.gpus.size())) {
        std::cerr << "Warning: Number of ranks (" << cfg.nranks
                  << ") exceeds number of GPUs (" << cfg.gpus.size()
                  << "). Multiple ranks will share GPUs." << std::endl;
    }

    // Generate coordination directory if not specified
    if (cfg.coord_dir.empty()) {
        cfg.coord_dir = "/tmp/rrun_" + generate_session_id();
    }

    // Default to "all" if --bind-to was not explicitly specified
    if (cfg.bind_state == BindToState::NotSpecified) {
        cfg.bind_cpu = true;
        cfg.bind_memory = true;
        cfg.bind_network = true;
    }

    // Discover system topology
    cucascade::memory::topology_discovery discovery;
    if (discovery.discover()) {
        cfg.topology = discovery.get_topology();
        // Build GPU ID to topology info mapping
        for (auto const& gpu : cfg.topology->gpus) {
            cfg.gpu_topology_map[static_cast<int>(gpu.id)] = &gpu;
        }
    } else {
        if (cfg.verbose) {
            std::cerr << "Warning: Failed to discover system topology. "
                      << "CPU affinity, NUMA binding, and UCX network device "
                      << "configuration will be skipped." << std::endl;
        }
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

        // Execute child body (should not return)
        child_body();
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
 * @brief Launch a single rank locally (fork-based).
 */
pid_t launch_rank_local(
    Config const& cfg, int rank, int* out_fd_stdout, int* out_fd_stderr
) {
    // Capture rank by value explicitly to avoid any potential issues
    int captured_rank = rank;
    return fork_with_piped_stdio(
        out_fd_stdout,
        out_fd_stderr,
        /*combine_stderr*/ false,
        [&cfg, captured_rank]() {
            // Set custom environment variables first (can be overridden by specific vars)
            for (auto const& env_pair : cfg.env_vars) {
                setenv(env_pair.first.c_str(), env_pair.second.c_str(), 1);
            }

            // Set environment variables
            setenv("RAPIDSMPF_RANK", std::to_string(captured_rank).c_str(), 1);
            setenv("RAPIDSMPF_NRANKS", std::to_string(cfg.nranks).c_str(), 1);
            setenv("RAPIDSMPF_COORD_DIR", cfg.coord_dir.c_str(), 1);

            // Set CUDA_VISIBLE_DEVICES if GPUs are available
            int gpu_id = -1;
            if (!cfg.gpus.empty()) {
                gpu_id = cfg.gpus[static_cast<size_t>(captured_rank) % cfg.gpus.size()];
                setenv("CUDA_VISIBLE_DEVICES", std::to_string(gpu_id).c_str(), 1);
            }

            // Apply topology-based configuration if available
            if (cfg.topology.has_value() && gpu_id >= 0) {
                auto it = cfg.gpu_topology_map.find(gpu_id);
                if (it != cfg.gpu_topology_map.end()) {
                    auto const& gpu_info = *it->second;

                    if (cfg.bind_cpu && !gpu_info.cpu_affinity_list.empty()) {
                        if (!set_cpu_affinity(gpu_info.cpu_affinity_list)) {
                            std::cerr << "Warning: Failed to set CPU affinity for rank "
                                      << captured_rank << " (GPU " << gpu_id << ")"
                                      << std::endl;
                        }
                    }

                    if (cfg.bind_memory && !gpu_info.memory_binding.empty()) {
                        if (!set_numa_memory_binding(gpu_info.memory_binding)) {
#if RAPIDSMPF_HAVE_NUMA
                            std::cerr
                                << "Warning: Failed to set NUMA memory binding for rank "
                                << captured_rank << " (GPU " << gpu_id << ")"
                                << std::endl;
#endif
                        }
                    }

                    if (cfg.bind_network && !gpu_info.network_devices.empty()) {
                        std::string ucx_net_devices;
                        for (size_t i = 0; i < gpu_info.network_devices.size(); ++i) {
                            if (i > 0) {
                                ucx_net_devices += ",";
                            }
                            ucx_net_devices += gpu_info.network_devices[i];
                        }
                        setenv("UCX_NET_DEVICES", ucx_net_devices.c_str(), 1);
                    }
                }
            }

            // Prepare arguments for execvp
            std::vector<char*> exec_args;
            exec_args.push_back(const_cast<char*>(cfg.app_binary.c_str()));
            for (auto const& arg : cfg.app_args) {
                exec_args.push_back(const_cast<char*>(arg.c_str()));
            }
            exec_args.push_back(nullptr);

            execvp(cfg.app_binary.c_str(), exec_args.data());
            std::cerr << "Failed to execute " << cfg.app_binary << ": "
                      << std::strerror(errno) << std::endl;
            _exit(1);
        }
    );
}

/**
 * @brief Wait for all child processes and check their exit status.
 */
int wait_for_ranks(std::vector<pid_t> const& pids) {
    int overall_status = 0;

    for (size_t i = 0; i < pids.size(); ++i) {
        int status;
        while (true) {
            pid_t result = waitpid(pids[i], &status, 0);

            if (result < 0) {
                if (errno == EINTR) {
                    // Retry waitpid for the same pid
                    continue;
                }
                std::cerr << "Error waiting for rank " << i << ": "
                          << std::strerror(errno) << std::endl;
                overall_status = 1;
                break;
            }

            if (WIFEXITED(status)) {
                int exit_code = WEXITSTATUS(status);
                if (exit_code != 0) {
                    std::cerr << "Rank " << i << " (PID " << pids[i]
                              << ") exited with code " << exit_code << std::endl;
                    overall_status = exit_code;
                }
            } else if (WIFSIGNALED(status)) {
                int signal = WTERMSIG(status);
                std::cerr << "Rank " << i << " (PID " << pids[i]
                          << ") terminated by signal " << signal << std::endl;
                overall_status = 128 + signal;
            }
            break;
        }
    }

    return overall_status;
}
}  // namespace

int main(int argc, char* argv[]) {
    try {
        // Parse arguments
        Config cfg = parse_args(argc, argv);

        if (cfg.verbose) {
            std::cout << "rrun configuration:\n";
            std::cout << "  Mode:          Single-node\n"
                      << "  GPUs:          ";
            if (cfg.gpus.empty()) {
                std::cout << "(none)\n";
            } else {
                for (size_t i = 0; i < cfg.gpus.size(); ++i) {
                    if (i > 0)
                        std::cout << ", ";
                    std::cout << cfg.gpus[i];
                }
                std::cout << "\n";
            }
            if (cfg.tag_output) {
                std::cout << "  Tag Output:    Yes\n";
            }
            std::cout << "  Ranks:         " << cfg.nranks << "\n"
                      << "  Application:   " << cfg.app_binary << "\n"
                      << "  Coord Dir:     " << cfg.coord_dir << "\n"
                      << "  Cleanup:       " << (cfg.cleanup ? "yes" : "no") << "\n";
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
                for (size_t i = 0; i < bind_types.size(); ++i) {
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

        std::filesystem::create_directories(cfg.coord_dir);

        std::vector<pid_t> pids;
        pids.reserve(static_cast<size_t>(cfg.nranks));

        // Block SIGINT/SIGTERM in this thread; a dedicated thread will handle them.
        sigset_t signal_set;
        sigemptyset(&signal_set);
        sigaddset(&signal_set, SIGINT);
        sigaddset(&signal_set, SIGTERM);
        sigprocmask(SIG_BLOCK, &signal_set, nullptr);

        // Output suppression flag and forwarder threads
        auto suppress_output = std::make_shared<std::atomic<bool>>(false);
        std::vector<std::thread> forwarders;
        forwarders.reserve(static_cast<size_t>(cfg.nranks) * 2);

        // Helper to start a forwarder thread for a given fd
        auto start_forwarder = [&](int fd, int rank, bool to_stderr) {
            if (fd < 0) {
                return;
            }
            forwarders.emplace_back([fd, rank, to_stderr, &cfg, suppress_output]() {
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
                        // Discard further lines after suppression
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

        // Single-node local mode
        for (int rank = 0; rank < cfg.nranks; ++rank) {
            int fd_out = -1;
            int fd_err = -1;
            pid_t pid = launch_rank_local(cfg, rank, &fd_out, &fd_err);
            pids.push_back(pid);

            if (cfg.verbose) {
                std::ostringstream msg;
                msg << "Launched rank " << rank << " (PID " << pid << ")";
                if (!cfg.gpus.empty()) {
                    msg << " on GPU "
                        << cfg.gpus[static_cast<size_t>(rank) % cfg.gpus.size()];
                }
                msg << std::endl;
                std::string msg_str = msg.str();

                std::cout << msg_str;
                std::cout.flush();
            }
            // Parent-side forwarders for local stdout and stderr
            start_forwarder(fd_out, rank, false);
            start_forwarder(fd_err, rank, true);
        }

        // Start a signal-waiting thread to forward signals.
        std::thread([signal_set, &pids, suppress_output]() mutable {
            for (;;) {
                int sig = 0;
                int rc = sigwait(&signal_set, &sig);
                if (rc != 0) {
                    return;
                }
                // Stop printing further output immediately
                suppress_output->store(true, std::memory_order_relaxed);
                // Forward signal to all local children
                for (pid_t pid : pids) {
                    std::ignore = kill(pid, sig);
                }
            }
        }).detach();

        if (cfg.verbose) {
            std::cout << "\nAll ranks launched. Waiting for completion...\n" << std::endl;
        }

        // Wait for all ranks to complete
        int exit_status = wait_for_ranks(pids);

        // Join forwarders before cleanup
        for (auto& th : forwarders) {
            if (th.joinable()) {
                th.join();
            }
        }

        if (cfg.cleanup) {
            if (cfg.verbose) {
                std::cout << "Cleaning up coordination directory: " << cfg.coord_dir
                          << std::endl;
            }
            std::error_code ec;
            std::filesystem::remove_all(cfg.coord_dir, ec);
            if (ec) {
                std::cerr << "Warning: Failed to cleanup directory: " << cfg.coord_dir
                          << ": " << ec.message() << std::endl;
            }
        } else if (cfg.verbose) {
            std::cout << "Coordination directory preserved: " << cfg.coord_dir
                      << std::endl;
        }

        if (cfg.verbose && exit_status == 0) {
            std::cout << "\nAll ranks completed successfully." << std::endl;
        }

        return exit_status;

    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "Run with -h or --help for usage information." << std::endl;
        return 1;
    }
}
