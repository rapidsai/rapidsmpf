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

#ifdef RAPIDSMPF_HAVE_SLURM
#include <pmix.h>
#endif

// Hex encoding for binary-safe address transmission
namespace {
std::string hex_encode(std::string const& input) {
    static constexpr const char* hex_chars = "0123456789abcdef";
    std::string result;
    result.reserve(input.size() * 2);
    for (char ch : input) {
        auto c = static_cast<unsigned char>(ch);
        result.push_back(hex_chars[c >> 4]);
        result.push_back(hex_chars[c & 0x0F]);
    }
    return result;
}

#ifdef RAPIDSMPF_HAVE_SLURM
std::string hex_decode(std::string const& input) {
    std::string result;
    result.reserve(input.size() / 2);
    for (size_t i = 0; i < input.size(); i += 2) {
        auto high = static_cast<unsigned char>(
            (input[i] >= 'a') ? (input[i] - 'a' + 10) : (input[i] - '0')
        );
        auto low = static_cast<unsigned char>(
            (input[i + 1] >= 'a') ? (input[i + 1] - 'a' + 10) : (input[i + 1] - '0')
        );
        result.push_back(static_cast<char>((high << 4) | low));
    }
    return result;
}
#endif

// Forward declarations of mode execution functions (defined later, outside namespace)
struct Config;
int execute_slurm_passthrough_mode(Config const& cfg);
int execute_single_node_mode(Config& cfg);
#ifdef RAPIDSMPF_HAVE_SLURM
int execute_slurm_hybrid_mode(Config& cfg);
std::string launch_rank0_and_get_address(
    Config const& cfg, std::string const& address_file, int total_ranks
);
std::string coordinate_root_address_via_pmix(
    std::optional<std::string> const& root_address_to_publish, bool verbose
);
#endif
int launch_ranks_fork_based(
    Config const& cfg,
    int rank_offset,
    int ranks_per_task,
    int total_ranks,
    std::string const& root_address,
    bool is_root_parent
);
pid_t launch_rank_local(
    Config const& cfg,
    int global_rank,
    int local_rank,
    int total_ranks,
    std::string const& root_address,
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
    bool slurm_mode{false};  // Running under Slurm (--slurm or auto-detected)
    int slurm_local_id{-1};  // Local rank within node (SLURM_LOCALID)
    int slurm_global_rank{-1};  // Global rank (SLURM_PROCID)
    int slurm_ntasks{-1};  // Total number of tasks (SLURM_NTASKS)
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
        << "                     1. Passthrough (no -n): Apply bindings and exec\n"
        << "                     2. Hybrid (with -n): Launch N ranks per Slurm task\n"
        << "                     In hybrid mode, each Slurm task launches multiple\n"
        << "                     ranks with coordinated global rank numbering\n\n"
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
        << "  # Passthrough mode (1 rank per Slurm task, 8 tasks total):\n"
        << "  srun --mpi=pmix -N 2 --ntasks-per-node=4 --gres=gpu:4 rrun "
           "./bench_shuffle -C ucxx\n\n"
        << "  # Hybrid mode (2 Slurm tasks × 4 ranks/task = 8 total ranks):\n"
        << "  srun --mpi=pmix -N 2 --ntasks-per-node=1 --gres=gpu:4 rrun -n 4 "
           "./bench_shuffle -C ucxx\n\n"
        << "  # Hybrid mode with --gpus-per-task:\n"
        << "  srun --mpi=pmix --ntasks-per-node=2 --gpus-per-task=4 rrun -n 4 "
           "./bench_shuffle -C ucxx\n\n"
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
 * @brief Check if running under Slurm and populate Slurm-related config fields.
 *
 * @param cfg Configuration to populate with Slurm information.
 * @return true if running under Slurm with required environment variables.
 */
bool detect_slurm_environment(Config& cfg) {
    // Check for required Slurm environment variables
    char const* slurm_job_id = std::getenv("SLURM_JOB_ID");
    char const* slurm_local_id = std::getenv("SLURM_LOCALID");
    char const* slurm_procid = std::getenv("SLURM_PROCID");
    char const* slurm_ntasks = std::getenv("SLURM_NTASKS");

    // Need at least job ID and local ID to be in Slurm mode
    if (!slurm_job_id || !slurm_local_id) {
        return false;
    }

    try {
        cfg.slurm_local_id = std::stoi(slurm_local_id);

        if (slurm_procid) {
            cfg.slurm_global_rank = std::stoi(slurm_procid);
        }

        if (slurm_ntasks) {
            cfg.slurm_ntasks = std::stoi(slurm_ntasks);
        } else {
            // Try SLURM_NPROCS as fallback
            char const* slurm_nprocs = std::getenv("SLURM_NPROCS");
            if (slurm_nprocs) {
                cfg.slurm_ntasks = std::stoi(slurm_nprocs);
            }
        }

        return true;
    } catch (...) {
        return false;
    }
}

/**
 * @brief Apply topology-based bindings for a specific GPU.
 *
 * This function sets CPU affinity, NUMA memory binding, and network device
 * environment variables based on the topology information for the given GPU.
 *
 * @param cfg Configuration containing topology information.
 * @param gpu_id GPU ID to apply bindings for.
 * @param verbose Print warnings on failure.
 */
void apply_topology_bindings(Config const& cfg, int gpu_id, bool verbose) {
    if (!cfg.topology.has_value() || gpu_id < 0) {
        return;
    }

    auto it = cfg.gpu_topology_map.find(gpu_id);
    if (it == cfg.gpu_topology_map.end()) {
        if (verbose) {
            std::cerr << "[rrun] Warning: No topology information for GPU " << gpu_id
                      << std::endl;
        }
        return;
    }

    auto const& gpu_info = *it->second;

    if (cfg.bind_cpu && !gpu_info.cpu_affinity_list.empty()) {
        if (!set_cpu_affinity(gpu_info.cpu_affinity_list)) {
            if (verbose) {
                std::cerr << "[rrun] Warning: Failed to set CPU affinity for GPU "
                          << gpu_id << std::endl;
            }
        }
    }

    if (cfg.bind_memory && !gpu_info.memory_binding.empty()) {
        if (!set_numa_memory_binding(gpu_info.memory_binding)) {
#if RAPIDSMPF_HAVE_NUMA
            if (verbose) {
                std::cerr << "[rrun] Warning: Failed to set NUMA memory binding for GPU "
                          << gpu_id << std::endl;
            }
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
        if (cfg.slurm_local_id < 0) {
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
            std::cerr << "[rrun] Warning: Failed to discover system topology. "
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
 * @param root_address Pre-coordinated root address (empty for FILE backend).
 * @param is_root_parent Whether this is root parent (affects launch logic).
 * @param coord_dir_hint Hint for coordination directory name (e.g., job ID).
 * @return Exit status (0 for success).
 */
int setup_launch_and_cleanup(
    Config& cfg,
    int rank_offset,
    int ranks_per_task,
    int total_ranks,
    std::string const& root_address,
    bool is_root_parent,
    std::string const& coord_dir_hint = ""
) {
    // Set up coordination directory
    if (cfg.coord_dir.empty()) {
        if (!coord_dir_hint.empty()) {
            cfg.coord_dir = "/tmp/rrun_" + coord_dir_hint;
        } else {
            cfg.coord_dir = "/tmp/rrun_" + generate_session_id();
        }
    }
    std::filesystem::create_directories(cfg.coord_dir);

    // Launch ranks and wait for completion
    int exit_status = launch_ranks_fork_based(
        cfg, rank_offset, ranks_per_task, total_ranks, root_address, is_root_parent
    );

    if (cfg.cleanup) {
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
    } else if (cfg.verbose) {
        std::cout << "[rrun] Coordination directory preserved: " << cfg.coord_dir
                  << std::endl;
    }

    if (cfg.verbose && exit_status == 0) {
        std::cout << "\n[rrun] All ranks completed successfully." << std::endl;
    }

    return exit_status;
}

#ifdef RAPIDSMPF_HAVE_SLURM
/**
 * @brief Execute application in Slurm passthrough mode (single rank per task).
 *
 * Applies topology bindings and executes the application directly without forking.
 *
 * @param cfg Configuration.
 * @return Exit status. Does not return on success, only on error.
 */
int execute_slurm_passthrough_mode(Config const& cfg) {
    if (cfg.verbose) {
        std::cout << "[rrun] Slurm passthrough mode: applying bindings and exec'ing"
                  << std::endl;
    }

    // Set rrun coordination environment variables so the application knows
    // it's being launched by rrun and should use bootstrap mode
    setenv("RAPIDSMPF_RANK", std::to_string(cfg.slurm_global_rank).c_str(), 1);
    setenv("RAPIDSMPF_NRANKS", std::to_string(cfg.slurm_ntasks).c_str(), 1);

    // Determine GPU for this Slurm task
    int gpu_id = -1;
    if (!cfg.gpus.empty()) {
        gpu_id = cfg.gpus[static_cast<size_t>(cfg.slurm_local_id) % cfg.gpus.size()];
        setenv("CUDA_VISIBLE_DEVICES", std::to_string(gpu_id).c_str(), 1);

        if (cfg.verbose) {
            std::cout << "[rrun] Slurm task (passthrough) local_id=" << cfg.slurm_local_id
                      << " assigned to GPU " << gpu_id << std::endl;
        }
    }

    // Set custom environment variables
    for (auto const& env_pair : cfg.env_vars) {
        setenv(env_pair.first.c_str(), env_pair.second.c_str(), 1);
    }

    apply_topology_bindings(cfg, gpu_id, cfg.verbose);

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
    return 1;
}

/**
 * @brief Execute application in Slurm hybrid mode with PMIx coordination.
 *
 * Root parent launches rank 0 first to get address, coordinates via PMIx, then parents
 * on all nodes launch their remaining ranks. Uses fork-based execution.
 *
 * @param cfg Configuration.
 * @return Exit status (0 for success).
 */
int execute_slurm_hybrid_mode(Config& cfg) {
    if (cfg.verbose) {
        std::cout << "[rrun] Slurm hybrid mode: task " << cfg.slurm_global_rank
                  << " launching " << cfg.nranks << " ranks per task" << std::endl;
        std::cout << "[rrun] Using PMIx for parent coordination (no file I/O)"
                  << std::endl;
    }

    // Set up coordination directory FIRST (needed by rank 0 when it's launched early)
    char const* job_id = std::getenv("SLURM_JOB_ID");
    if (cfg.coord_dir.empty()) {
        if (job_id) {
            cfg.coord_dir = "/tmp/rrun_slurm_" + std::string{job_id};
        } else {
            cfg.coord_dir = "/tmp/rrun_" + generate_session_id();
        }
    }
    std::filesystem::create_directories(cfg.coord_dir);

    // Root parent needs to launch rank 0 first to get address
    bool is_root_parent = (cfg.slurm_global_rank == 0);

    // Coordinate root address with other nodes via PMIx
    int slurm_ntasks = cfg.slurm_ntasks > 0 ? cfg.slurm_ntasks : 1;
    int total_ranks = slurm_ntasks * cfg.nranks;
    std::string coordinated_root_address;

    if (is_root_parent) {
        // Root parent: Launch rank 0, get address, coordinate via PMIx
        std::string address_file =
            "/tmp/rapidsmpf_root_address_" + std::string{job_id ? job_id : "unknown"};
        coordinated_root_address =
            launch_rank0_and_get_address(cfg, address_file, total_ranks);
        coordinated_root_address =
            coordinate_root_address_via_pmix(coordinated_root_address, cfg.verbose);
    } else {
        // Non-root parent: Get address from root via PMIx
        coordinated_root_address =
            coordinate_root_address_via_pmix(std::nullopt, cfg.verbose);
    }

    // Now all parents have the coordinated_root_address
    // Continue to fork-based launch below with this address
    unsetenv("RAPIDSMPF_ROOT_ADDRESS_FILE");

    // Calculate rank offsets
    int rank_offset = cfg.slurm_global_rank * cfg.nranks;

    if (cfg.verbose) {
        std::cout << "[rrun] Task " << cfg.slurm_global_rank << " launching ranks "
                  << rank_offset << "-" << (rank_offset + cfg.nranks - 1)
                  << " (total: " << total_ranks << " ranks)" << std::endl;
    }

    // Use common helper for launch and cleanup
    std::string coord_hint = job_id ? ("slurm_" + std::string{job_id}) : "";
    int exit_status = setup_launch_and_cleanup(
        cfg,
        rank_offset,
        cfg.nranks,
        total_ranks,
        coordinated_root_address,
        is_root_parent,
        coord_hint
    );

    // Finalize PMIx
    if (!coordinated_root_address.empty()) {
        if (cfg.verbose) {
            std::cout << "[rrun] Finalizing PMIx in parent" << std::endl;
        }
        PMIx_Finalize(nullptr, 0);
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

    // Use common helper for launch and cleanup
    // rank_offset=0, ranks_per_task=nranks, total_ranks=nranks, no root_address, not
    // root_parent
    return setup_launch_and_cleanup(cfg, 0, cfg.nranks, cfg.nranks, "", false);
}

#ifdef RAPIDSMPF_HAVE_SLURM
/**
 * @brief Launch rank 0 first to obtain its UCXX root address.
 *
 * @param cfg Configuration.
 * @param address_file Path to file where rank 0 will write its address.
 * @param total_ranks Total number of ranks across all tasks.
 * @return Hex-encoded root address.
 *
 * @throws std::runtime_error on timeout or launch failure.
 */
std::string launch_rank0_and_get_address(
    Config const& cfg, std::string const& address_file, int total_ranks
) {
    if (cfg.verbose) {
        std::cout << "[rrun] Root parent: launching rank 0 first to get address"
                  << std::endl;
    }

    setenv("RAPIDSMPF_ROOT_ADDRESS_FILE", address_file.c_str(), 1);

    int fd_out = -1, fd_err = -1;
    pid_t rank0_pid = launch_rank_local(cfg, 0, 0, total_ranks, "", &fd_out, &fd_err);

    // Start forwarders for rank 0 output
    std::thread rank0_stdout_forwarder;
    std::thread rank0_stderr_forwarder;
    auto suppress = std::make_shared<std::atomic<bool>>(false);

    if (fd_out >= 0) {
        rank0_stdout_forwarder = std::thread([fd_out, suppress]() {
            FILE* stream = fdopen(fd_out, "r");
            if (!stream) {
                close(fd_out);
                return;
            }
            char buffer[4096];
            while (fgets(buffer, sizeof(buffer), stream) != nullptr) {
                if (suppress->load())
                    continue;
                std::lock_guard<std::mutex> lock(output_mutex);
                fputs(buffer, stdout);
                fflush(stdout);
            }
            fclose(stream);
        });
    }

    if (fd_err >= 0) {
        rank0_stderr_forwarder = std::thread([fd_err, suppress]() {
            FILE* stream = fdopen(fd_err, "r");
            if (!stream) {
                close(fd_err);
                return;
            }
            char buffer[4096];
            while (fgets(buffer, sizeof(buffer), stream) != nullptr) {
                if (suppress->load())
                    continue;
                std::lock_guard<std::mutex> lock(output_mutex);
                fputs(buffer, stderr);
                fflush(stderr);
            }
            fclose(stream);
        });
    }

    // Wait for rank 0 to write the address file (with timeout)
    auto start = std::chrono::steady_clock::now();
    while (!std::filesystem::exists(address_file)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed > std::chrono::seconds(30)) {
            suppress->store(true);
            kill(rank0_pid, SIGKILL);
            waitpid(rank0_pid, nullptr, 0);
            if (rank0_stdout_forwarder.joinable())
                rank0_stdout_forwarder.join();
            if (rank0_stderr_forwarder.joinable())
                rank0_stderr_forwarder.join();
            throw std::runtime_error("Timeout waiting for rank 0 to write root address");
        }
    }

    // Read the hex-encoded address, decode and remove file
    std::string encoded_address;
    std::ifstream addr_stream(address_file);
    std::getline(addr_stream, encoded_address);
    addr_stream.close();
    std::string root_address = hex_decode(encoded_address);
    std::filesystem::remove(address_file);

    if (cfg.verbose) {
        std::cout << "[rrun] Got root address from rank 0 (hex-encoded, "
                  << encoded_address.size() << " chars -> " << root_address.size()
                  << " bytes)" << std::endl;
    }

    // Rank 0 is already running - detach forwarders
    if (rank0_stdout_forwarder.joinable())
        rank0_stdout_forwarder.detach();
    if (rank0_stderr_forwarder.joinable())
        rank0_stderr_forwarder.detach();

    return root_address;
}

/**
 * @brief Coordinate root address between parent processes using PMIx.
 *
 * This function is called by parent rrun processes in Slurm hybrid mode.
 * The root parent (PMIX_RANK=0) publishes the root address, and non-root
 * parents retrieve it. This avoids file-based coordination.
 *
 * @param root_address_to_publish Root address to publish. If set (has_value()), this is
 *                                the root parent and it will publish. If empty (nullopt),
 *                                this is a non-root parent and it will retrieve.
 * @param verbose Whether to print debug messages.
 * @return Root address (either published or retrieved).
 *
 * @throws std::runtime_error on PMIx errors.
 */
std::string coordinate_root_address_via_pmix(
    std::optional<std::string> const& root_address_to_publish, bool verbose
) {
    // Initialize PMIx for parent process
    pmix_proc_t proc;
    pmix_status_t rc = PMIx_Init(&proc, nullptr, 0);
    if (rc != PMIX_SUCCESS) {
        throw std::runtime_error(
            "PMIx_Init failed in rrun parent: " + std::string{PMIx_Error_string(rc)}
        );
    }

    if (verbose) {
        std::cout << "[rrun] Parent PMIx initialized: rank " << proc.rank
                  << ", namespace " << proc.nspace << std::endl;
    }

    std::string root_address;

    if (root_address_to_publish.has_value()) {
        // Root parent publishes the address (hex-encoded for binary safety)
        std::string encoded_address = hex_encode(root_address_to_publish.value());

        if (verbose) {
            std::cout << "[rrun] Publishing root address via PMIx (hex-encoded, "
                      << root_address_to_publish.value().size() << " bytes -> "
                      << encoded_address.size() << " chars)" << std::endl;
        }

        // Use PMIx_Put with GLOBAL scope
        pmix_value_t value;
        PMIX_VALUE_CONSTRUCT(&value);
        value.type = PMIX_STRING;
        value.data.string = strdup(encoded_address.c_str());

        rc = PMIx_Put(PMIX_GLOBAL, "rapidsmpf_root_address", &value);
        PMIX_VALUE_DESTRUCT(&value);

        if (rc != PMIX_SUCCESS) {
            PMIx_Finalize(nullptr, 0);
            throw std::runtime_error(
                "PMIx_Put failed: " + std::string{PMIx_Error_string(rc)}
            );
        }

        // Commit the data
        rc = PMIx_Commit();
        if (rc != PMIX_SUCCESS) {
            PMIx_Finalize(nullptr, 0);
            throw std::runtime_error(
                "PMIx_Commit failed: " + std::string{PMIx_Error_string(rc)}
            );
        }

        root_address = root_address_to_publish.value();
    }

    // Barrier with PMIX_COLLECT_DATA to ensure data exchange
    pmix_info_t info;
    PMIX_INFO_CONSTRUCT(&info);
    bool collect_data = true;
    PMIX_INFO_LOAD(&info, PMIX_COLLECT_DATA, &collect_data, PMIX_BOOL);

    pmix_proc_t proc_wildcard;
    PMIX_PROC_CONSTRUCT(&proc_wildcard);
    std::memcpy(proc_wildcard.nspace, proc.nspace, PMIX_MAX_NSLEN + 1);
    proc_wildcard.rank = PMIX_RANK_WILDCARD;

    rc = PMIx_Fence(&proc_wildcard, 1, &info, 1);
    PMIX_INFO_DESTRUCT(&info);

    // Accept partial success (some PMIx implementations return this for fences)
    if (rc != PMIX_SUCCESS && rc != PMIX_ERR_PARTIAL_SUCCESS) {
        PMIx_Finalize(nullptr, 0);
        throw std::runtime_error(
            "PMIx_Fence failed: " + std::string{PMIx_Error_string(rc)}
        );
    }

    if (!root_address_to_publish.has_value()) {
        // Non-root parents retrieve the address
        pmix_proc_t source_proc;
        PMIX_PROC_CONSTRUCT(&source_proc);
        std::memcpy(source_proc.nspace, proc.nspace, PMIX_MAX_NSLEN + 1);
        source_proc.rank = 0;  // Get from rank 0

        pmix_value_t* value = nullptr;
        rc = PMIx_Get(&source_proc, "rapidsmpf_root_address", nullptr, 0, &value);

        if (rc != PMIX_SUCCESS || value == nullptr) {
            PMIx_Finalize(nullptr, 0);
            throw std::runtime_error(
                "PMIx_Get failed: " + std::string{PMIx_Error_string(rc)}
            );
        }

        if (value->type != PMIX_STRING) {
            PMIX_VALUE_RELEASE(value);
            PMIx_Finalize(nullptr, 0);
            throw std::runtime_error("PMIx_Get returned non-string value");
        }

        std::string encoded_address = value->data.string;
        PMIX_VALUE_RELEASE(value);

        root_address = hex_decode(encoded_address);

        if (verbose) {
            std::cout << "[rrun] Retrieved root address via PMIx (hex-encoded, "
                      << encoded_address.size() << " chars -> " << root_address.size()
                      << " bytes)" << std::endl;
        }
    }

    // Keep PMIx session alive - will finalize after children complete
    // Note: We don't call PMIx_Finalize here because we want the session
    // to stay alive while children are running

    return root_address;
}
#endif  // RAPIDSMPF_HAVE_SLURM

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
 * @param root_address Pre-coordinated root address (empty for FILE backend).
 * @param is_root_parent Whether this is root parent (affects which ranks to launch).
 * @return Exit status (0 for success).
 */
int launch_ranks_fork_based(
    Config const& cfg,
    int rank_offset,
    int ranks_per_task,
    int total_ranks,
    std::string const& root_address,
    bool is_root_parent
) {
    std::vector<pid_t> pids;
    pids.reserve(static_cast<size_t>(ranks_per_task));

    // Block SIGINT/SIGTERM in this thread; a dedicated thread will handle them.
    sigset_t signal_set;
    sigemptyset(&signal_set);
    sigaddset(&signal_set, SIGINT);
    sigaddset(&signal_set, SIGTERM);
    sigprocmask(SIG_BLOCK, &signal_set, nullptr);

    // Output suppression flag and forwarder threads
    auto suppress_output = std::make_shared<std::atomic<bool>>(false);
    std::vector<std::thread> forwarders;
    forwarders.reserve(static_cast<size_t>(ranks_per_task) * 2);

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

    // Launch ranks (skip rank 0 if root parent already launched it)
    int start_local_rank = (is_root_parent && !root_address.empty()) ? 1 : 0;

    for (int local_rank = start_local_rank; local_rank < ranks_per_task; ++local_rank) {
        int global_rank = rank_offset + local_rank;
        int fd_out = -1;
        int fd_err = -1;
        pid_t pid = launch_rank_local(
            cfg, global_rank, local_rank, total_ranks, root_address, &fd_out, &fd_err
        );
        pids.push_back(pid);

        if (cfg.verbose) {
            std::ostringstream msg;
            msg << "[rrun] Launched rank " << global_rank << " (PID " << pid << ")";
            if (!cfg.gpus.empty()) {
                msg << " on GPU "
                    << cfg.gpus[static_cast<size_t>(local_rank) % cfg.gpus.size()];
            }
            msg << std::endl;
            std::string msg_str = msg.str();

            std::cout << msg_str;
            std::cout.flush();
        }
        start_forwarder(fd_out, global_rank, false);
        start_forwarder(fd_err, global_rank, true);
    }

    // Start a signal-waiting thread to forward signals.
    std::thread([signal_set, &pids, suppress_output]() mutable {
        for (;;) {
            int sig = 0;
            int rc = sigwait(&signal_set, &sig);
            if (rc != 0) {
                continue;
            }
            suppress_output->store(true, std::memory_order_relaxed);
            for (pid_t pid : pids) {
                kill(pid, sig);
            }
            return;
        }
    }).detach();

    std::cout << "\n[rrun] All ranks launched. Waiting for completion...\n" << std::endl;

    // Wait for all processes
    int exit_status = 0;
    for (size_t i = 0; i < pids.size(); ++i) {
        int status = 0;
        pid_t pid = pids[i];
        if (waitpid(pid, &status, 0) < 0) {
            std::cerr << "[rrun] Failed to wait for rank " << i << " (PID " << pid
                      << "): " << std::strerror(errno) << std::endl;
            exit_status = 1;
            continue;
        }

        if (WIFEXITED(status)) {
            int code = WEXITSTATUS(status);
            if (code != 0) {
                std::cerr << "[rrun] Rank "
                          << (static_cast<size_t>(rank_offset)
                              + (is_root_parent && !root_address.empty() ? i + 1 : i))
                          << " (PID " << pid << ") exited with code " << code
                          << std::endl;
                exit_status = code;
            }
        } else if (WIFSIGNALED(status)) {
            int sig = WTERMSIG(status);
            std::cerr << "[rrun] Rank "
                      << (static_cast<size_t>(rank_offset)
                          + (is_root_parent && !root_address.empty() ? i + 1 : i))
                      << " (PID " << pid << ") terminated by signal " << sig << std::endl;
            exit_status = 128 + sig;
        }
    }

    // Wait for forwarder threads to finish
    for (auto& t : forwarders) {
        if (t.joinable()) {
            t.join();
        }
    }

    return exit_status;
}

/**
 * @brief Launch a single rank locally (fork-based).
 *
 * @param cfg Configuration.
 * @param global_rank Global rank number (used for RAPIDSMPF_RANK).
 * @param local_rank Local rank for GPU assignment (defaults to global_rank).
 * @param total_ranks Total number of ranks across all tasks (used for RAPIDSMPF_NRANKS).
 * @param root_address Optional pre-coordinated root address (for hybrid mode).
 * @param out_fd_stdout Output file descriptor for stdout.
 * @param out_fd_stderr Output file descriptor for stderr.
 * @return Child process PID.
 */
pid_t launch_rank_local(
    Config const& cfg,
    int global_rank,
    int local_rank,
    int total_ranks,
    std::string const& root_address,
    int* out_fd_stdout,
    int* out_fd_stderr
) {
    // Capture all parameters by value to avoid any potential issues
    int captured_global_rank = global_rank;
    int captured_local_rank = local_rank;
    int captured_total_ranks = total_ranks;
    std::string captured_root_address = root_address;

    return fork_with_piped_stdio(
        out_fd_stdout,
        out_fd_stderr,
        /*combine_stderr*/ false,
        [&cfg,
         captured_global_rank,
         captured_local_rank,
         captured_total_ranks,
         captured_root_address]() {
            // Set custom environment variables first (can be overridden by specific vars)
            for (auto const& env_pair : cfg.env_vars) {
                setenv(env_pair.first.c_str(), env_pair.second.c_str(), 1);
            }

            // Set environment variables
            setenv("RAPIDSMPF_RANK", std::to_string(captured_global_rank).c_str(), 1);
            setenv("RAPIDSMPF_NRANKS", std::to_string(captured_total_ranks).c_str(), 1);

            // Always set coord_dir for bootstrap initialization
            // (needed even if using RAPIDSMPF_ROOT_ADDRESS for coordination)
            if (!cfg.coord_dir.empty()) {
                setenv("RAPIDSMPF_COORD_DIR", cfg.coord_dir.c_str(), 1);
            }

            // If root address was pre-coordinated by parent, set it (hex-encoded)
            // This allows children to skip bootstrap coordination entirely
            if (!captured_root_address.empty()) {
                std::string encoded_address = hex_encode(captured_root_address);
                setenv("RAPIDSMPF_ROOT_ADDRESS", encoded_address.c_str(), 1);
            }

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
                gpu_id =
                    cfg.gpus[static_cast<size_t>(captured_local_rank) % cfg.gpus.size()];
                setenv("CUDA_VISIBLE_DEVICES", std::to_string(gpu_id).c_str(), 1);
            }

            apply_topology_bindings(cfg, gpu_id, cfg.verbose);

            // Prepare arguments for execvp
            std::vector<char*> exec_args;
            exec_args.push_back(const_cast<char*>(cfg.app_binary.c_str()));
            for (auto const& arg : cfg.app_args) {
                exec_args.push_back(const_cast<char*>(arg.c_str()));
            }
            exec_args.push_back(nullptr);

            execvp(cfg.app_binary.c_str(), exec_args.data());
            std::cerr << "[rrun] Failed to execute " << cfg.app_binary << ": "
                      << std::strerror(errno) << std::endl;
            _exit(1);
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
                for (size_t i = 0; i < cfg.gpus.size(); ++i) {
                    if (i > 0)
                        std::cout << ", ";
                    std::cout << cfg.gpus[i];
                }
                std::cout << "\n";
            }
            if (cfg.slurm_mode) {
                std::cout << "  Slurm Local ID: " << cfg.slurm_local_id << "\n"
                          << "  Slurm Rank:     " << cfg.slurm_global_rank << "\n"
                          << "  Slurm NTasks:   " << cfg.slurm_ntasks << "\n";
            } else {
                if (cfg.tag_output) {
                    std::cout << "  Tag Output:    Yes\n";
                }
                std::cout << "  Ranks:         " << cfg.nranks << "\n"
                          << "  Coord Dir:     " << cfg.coord_dir << "\n"
                          << "  Cleanup:       " << (cfg.cleanup ? "yes" : "no") << "\n";
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

        if (cfg.slurm_mode) {
            if (cfg.nranks == 1) {
                // Slurm passthrough mode: single rank per task, no forking
                return execute_slurm_passthrough_mode(cfg);
            } else {
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
            }
        } else {
            // Single-node mode with FILE backend
            return execute_single_node_mode(cfg);
        }

    } catch (std::exception const& e) {
        std::cerr << "[rrun] Error: " << e.what() << std::endl;
        std::cerr << "[rrun] Run with -h or --help for usage information." << std::endl;
        return 1;
    }
}
