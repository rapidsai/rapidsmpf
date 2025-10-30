/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

/**
 * @brief Host information for multi-node deployment.
 */
struct HostInfo {
    std::string hostname;
    int slots{1};  // Number of processes to launch on this host
    std::vector<int> gpus;  // GPU IDs available on this host
};

/**
 * @brief Configuration for the rrun launcher.
 */
struct Config {
    int nranks{1};  // Total number of ranks
    int ppn{-1};  // Processes per node (-1 = auto from hostfile)
    std::string app_binary;  // Application binary path
    std::vector<std::string> app_args;  // Arguments to pass to application
    std::vector<int> gpus;  // GPU IDs to use (single-node mode)
    std::vector<HostInfo> hosts;  // Host list for multi-node mode
    std::string hostfile;  // Path to hostfile
    std::string coord_dir;  // Coordination directory
    std::string ssh_opts;  // Additional SSH options
    std::map<std::string, std::string> env_vars;  // Environment variables to pass
    bool verbose{false};  // Verbose output
    bool cleanup{true};  // Cleanup coordination directory on exit
    bool use_ssh{false};  // Multi-node mode via SSH
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
 * Currently using nvidia-smi to simplify multi-node detection. This will be replaced with
 * NVML in the future.
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
 * @brief Parse hostfile in OpenMPI/Slurm format.
 *
 * Format examples:
 *   node1 slots=4 gpus=0,1,2,3
 *   node2 slots=4
 *   node3
 */
std::vector<HostInfo> parse_hostfile(std::string const& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open hostfile: " + path);
    }

    std::vector<HostInfo> hosts;
    std::string line;
    int line_num = 0;

    while (std::getline(file, line)) {
        ++line_num;

        // Remove comments
        auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        // Skip empty lines
        if (line.empty())
            continue;

        HostInfo host;
        std::istringstream iss(line);
        iss >> host.hostname;

        if (host.hostname.empty()) {
            std::cerr << "Warning: Empty hostname on line " << line_num << std::endl;
            continue;
        }

        // Parse key=value pairs
        std::string token;
        while (iss >> token) {
            auto eq_pos = token.find('=');
            if (eq_pos == std::string::npos) {
                std::cerr << "Warning: Invalid token '" << token << "' on line "
                          << line_num << std::endl;
                continue;
            }

            std::string key = token.substr(0, eq_pos);
            std::string value = token.substr(eq_pos + 1);

            if (key == "slots") {
                try {
                    host.slots = std::stoi(value);
                } catch (...) {
                    throw std::runtime_error(
                        "Invalid slots value on line " + std::to_string(line_num)
                    );
                }
            } else if (key == "gpus") {
                // Parse comma-separated GPU list
                std::istringstream gpu_stream(value);
                std::string gpu_id;
                while (std::getline(gpu_stream, gpu_id, ',')) {
                    try {
                        host.gpus.push_back(std::stoi(gpu_id));
                    } catch (...) {
                        throw std::runtime_error(
                            "Invalid GPU ID '" + gpu_id + "' on line "
                            + std::to_string(line_num)
                        );
                    }
                }
            }
        }

        hosts.push_back(host);
    }

    if (hosts.empty()) {
        throw std::runtime_error("No valid hosts found in hostfile: " + path);
    }

    return hosts;
}

/**
 * @brief Create coordination directory.
 */
void create_coord_dir(std::string const& path) {
    if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        throw std::runtime_error(
            "Failed to create coordination directory: " + path + ": "
            + std::strerror(errno)
        );
    }
}

/**
 * @brief Recursively remove directory.
 */
void remove_dir_recursive(std::string const& path) {
    // Use system command for simplicity (rm -rf)
    std::string cmd = "rm -rf " + path;
    if (system(cmd.c_str()) != 0) {
        std::cerr << "Warning: Failed to cleanup directory: " << path << std::endl;
    }
}

/**
 * @brief Print usage information.
 */
void print_usage(char const* prog_name) {
    std::cout
        << "rrun - RapidsMPF Process Launcher\n\n"
        << "Usage: " << prog_name << " [options] <application> [app_args...]\n\n"
        << "Single-Node Options:\n"
        << "  -n <nranks>        Number of ranks to launch (required)\n"
        << "  -g <gpu_list>      Comma-separated list of GPU IDs (e.g., 0,1,2,3)\n"
        << "                     If not specified, auto-detect available GPUs\n\n"
        << "Multi-Node Options:\n"
        << "  --hostfile <file>  Path to hostfile (enables SSH mode)\n"
        << "  --ppn <num>        Processes per node (default: from hostfile slots)\n"
        << "  --ssh-opts <opts>  Additional SSH options (e.g., '-i ~/.ssh/key')\n\n"
        << "Common Options:\n"
        << "  -d <coord_dir>     Coordination directory (default: "
           "/tmp/rrun_<random>)\n"
        << "                     Must be on shared filesystem for multi-node\n"
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
        << "  rrun -n 2 ./bench_comm -C ucxx-bootstrap -O all-to-all\n\n"
        << "  # Launch 4 ranks on specific GPUs:\n"
        << "  rrun -n 4 -g 0,1,2,3 ./bench_comm -C ucxx-bootstrap\n\n"
        << "  # Launch with custom environment variables:\n"
        << "  rrun -n 2 -x UCX_TLS=cuda_copy,cuda_ipc,rc,tcp -x MY_VAR=value "
           "./bench_comm\n\n"
        << "Multi-Node Examples:\n"
        << "  # Launch using hostfile:\n"
        << "  rrun --hostfile hosts.txt ./bench_comm -C ucxx-bootstrap\n\n"
        << "  # Launch 2 processes per node:\n"
        << "  rrun --hostfile hosts.txt --ppn 2 ./bench_comm -C ucxx-bootstrap\n\n"
        << "  # Use specific coordination directory:\n"
        << "  rrun --hostfile hosts.txt -d /shared/nfs/coord ./bench_comm\n\n"
        << "Hostfile Format:\n"
        << "  node1 slots=4 gpus=0,1,2,3\n"
        << "  node2 slots=4 gpus=0,1,2,3\n"
        << "  # Lines starting with # are comments\n"
        << std::endl;
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
        } else if (arg == "--hostfile") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing argument for --hostfile");
            }
            cfg.hostfile = argv[++i];
            cfg.use_ssh = true;
        } else if (arg == "--ppn") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing argument for --ppn");
            }
            cfg.ppn = std::stoi(argv[++i]);
            if (cfg.ppn <= 0) {
                throw std::runtime_error("Invalid ppn: " + std::to_string(cfg.ppn));
            }
        } else if (arg == "--ssh-opts") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing argument for --ssh-opts");
            }
            cfg.ssh_opts = argv[++i];
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

    if (cfg.use_ssh) {
        // Parse hostfile if in SSH mode
        if (cfg.hostfile.empty()) {
            throw std::runtime_error("--hostfile required for multi-node mode");
        }
        cfg.hosts = parse_hostfile(cfg.hostfile);

        // Calculate total ranks from hostfile if not specified
        if (cfg.nranks == 1) {  // Default value, not explicitly set
            if (cfg.ppn > 0) {
                // ppn specified, calculate from hosts * ppn
                cfg.nranks = cfg.ppn * static_cast<int>(cfg.hosts.size());
            } else {
                // Use slots from hostfile
                cfg.nranks = 0;
                for (auto const& host : cfg.hosts) {
                    cfg.nranks += host.slots;
                }
            }
        }

        // Validate coordination directory for multi-node
        if (cfg.coord_dir.empty()) {
            std::cerr << "Warning: No coordination directory specified for multi-node.\n"
                      << "Using /tmp/rrun_<id> - ensure this is on a shared filesystem!"
                      << std::endl;
        }
    } else {
        // Single-node mode
        if (cfg.nranks <= 0) {
            throw std::runtime_error(
                "Number of ranks (-n) must be specified and positive"
            );
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
    }

    // Generate coordination directory if not specified
    if (cfg.coord_dir.empty()) {
        cfg.coord_dir = "/tmp/rrun_" + generate_session_id();
    }

    return cfg;
}

/**
 * @brief Launch a single rank locally (fork-based).
 */
pid_t launch_rank_local(Config const& cfg, int rank) {
    pid_t pid = fork();

    if (pid < 0) {
        throw std::runtime_error("Failed to fork: " + std::string{std::strerror(errno)});
    } else if (pid == 0) {
        // Child process

        // Preserve parent's LD_LIBRARY_PATH (important for development builds)
        // No need to set it explicitly as it's inherited from parent

        // Set custom environment variables first (can be overridden by specific vars)
        for (auto const& env_pair : cfg.env_vars) {
            setenv(env_pair.first.c_str(), env_pair.second.c_str(), 1);
        }

        // Set environment variables
        setenv("RAPIDSMPF_RANK", std::to_string(rank).c_str(), 1);
        setenv("RAPIDSMPF_NRANKS", std::to_string(cfg.nranks).c_str(), 1);
        setenv("RAPIDSMPF_COORD_DIR", cfg.coord_dir.c_str(), 1);

        // Set CUDA_VISIBLE_DEVICES if GPUs are available
        if (!cfg.gpus.empty()) {
            int gpu_id = cfg.gpus[static_cast<size_t>(rank) % cfg.gpus.size()];
            setenv("CUDA_VISIBLE_DEVICES", std::to_string(gpu_id).c_str(), 1);
        }

        // Prepare arguments for execvp
        std::vector<char*> exec_args;
        exec_args.push_back(const_cast<char*>(cfg.app_binary.c_str()));
        for (auto const& arg : cfg.app_args) {
            exec_args.push_back(const_cast<char*>(arg.c_str()));
        }
        exec_args.push_back(nullptr);

        // Execute application
        execvp(cfg.app_binary.c_str(), exec_args.data());

        // If execvp returns, it failed
        std::cerr << "Failed to execute " << cfg.app_binary << ": "
                  << std::strerror(errno) << std::endl;
        exit(1);
    }

    // Parent PID
    return pid;
}

/**
 * @brief Launch a single rank on a remote host via SSH.
 */
pid_t launch_rank_ssh(
    Config const& cfg,
    int rank,
    std::string const& hostname,
    int local_rank,
    std::vector<int> const& host_gpus
) {
    pid_t pid = fork();

    if (pid < 0) {
        throw std::runtime_error("Failed to fork: " + std::string{std::strerror(errno)});
    } else if (pid == 0) {
        // Child process - execute SSH command

        // Build the remote command
        std::ostringstream remote_cmd;

        // Set custom environment variables first
        for (auto const& env_pair : cfg.env_vars) {
            remote_cmd << env_pair.first << "=";
            // Quote value if it contains spaces or special characters
            if (env_pair.second.find(' ') != std::string::npos
                || env_pair.second.find('"') != std::string::npos
                || env_pair.second.find('\'') != std::string::npos)
            {
                // Escape double quotes and wrap in double quotes
                std::string escaped_value = env_pair.second;
                size_t pos = 0;
                while ((pos = escaped_value.find('"', pos)) != std::string::npos) {
                    escaped_value.insert(pos, "\\");
                    pos += 2;
                }
                remote_cmd << "\"" << escaped_value << "\" ";
            } else {
                remote_cmd << env_pair.second << " ";
            }
        }

        // Set environment variables
        remote_cmd << "RAPIDSMPF_RANK=" << rank << " ";
        remote_cmd << "RAPIDSMPF_NRANKS=" << cfg.nranks << " ";
        remote_cmd << "RAPIDSMPF_COORD_DIR=" << cfg.coord_dir << " ";

        // Set CUDA_VISIBLE_DEVICES if GPUs specified for this host
        if (!host_gpus.empty()) {
            int gpu_id = host_gpus[static_cast<size_t>(local_rank) % host_gpus.size()];
            remote_cmd << "CUDA_VISIBLE_DEVICES=" << gpu_id << " ";
        }

        // Add the application binary and arguments
        remote_cmd << cfg.app_binary;
        for (auto const& arg : cfg.app_args) {
            // Escape arguments that might contain spaces or special characters
            if (arg.find(' ') != std::string::npos) {
                remote_cmd << " \"" << arg << "\"";
            } else {
                remote_cmd << " " << arg;
            }
        }

        // Build SSH command
        std::vector<char*> ssh_args;
        ssh_args.push_back(const_cast<char*>("ssh"));

        // Add SSH options if provided
        if (!cfg.ssh_opts.empty()) {
            // Simple parsing - split on spaces
            std::istringstream iss(cfg.ssh_opts);
            static std::vector<std::string> opts_storage;
            std::string opt;
            while (iss >> opt) {
                opts_storage.push_back(opt);
            }
            for (auto const& opt : opts_storage) {
                ssh_args.push_back(const_cast<char*>(opt.c_str()));
            }
        }

        // Add hostname and remote command
        static std::string hostname_storage = hostname;
        static std::string remote_cmd_storage = remote_cmd.str();
        ssh_args.push_back(const_cast<char*>(hostname_storage.c_str()));
        ssh_args.push_back(const_cast<char*>(remote_cmd_storage.c_str()));
        ssh_args.push_back(nullptr);

        // Execute SSH
        execvp("ssh", ssh_args.data());

        // If execvp returns, it failed
        std::cerr << "Failed to execute ssh: " << std::strerror(errno) << std::endl;
        exit(1);
    }

    // Parent process
    return pid;
}

/**
 * @brief Wait for all child processes and check their exit status.
 */
int wait_for_ranks(std::vector<pid_t> const& pids) {
    int overall_status = 0;

    for (size_t i = 0; i < pids.size(); ++i) {
        int status;
        pid_t result = waitpid(pids[i], &status, 0);

        if (result < 0) {
            std::cerr << "Error waiting for rank " << i << ": " << std::strerror(errno)
                      << std::endl;
            overall_status = 1;
            continue;
        }

        if (WIFEXITED(status)) {
            int exit_code = WEXITSTATUS(status);
            if (exit_code != 0) {
                std::cerr << "Rank " << i << " (PID " << pids[i] << ") exited with code "
                          << exit_code << std::endl;
                overall_status = exit_code;
            }
        } else if (WIFSIGNALED(status)) {
            int signal = WTERMSIG(status);
            std::cerr << "Rank " << i << " (PID " << pids[i] << ") terminated by signal "
                      << signal << std::endl;
            overall_status = 128 + signal;
        }
    }

    return overall_status;
}

/**
 * @brief Signal handler to cleanup on interrupt.
 */
std::vector<pid_t>* g_child_pids = nullptr;

void signal_handler(int signum) {
    if (g_child_pids != nullptr) {
        // Forward signal to all children
        for (pid_t pid : *g_child_pids) {
            kill(pid, signum);
        }
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        // Parse arguments
        Config cfg = parse_args(argc, argv);

        if (cfg.verbose) {
            std::cout << "rrun configuration:\n";
            if (cfg.use_ssh) {
                std::cout << "  Mode:          Multi-node (SSH)\n"
                          << "  Hosts:         " << cfg.hosts.size() << "\n";
                for (size_t i = 0; i < cfg.hosts.size(); ++i) {
                    std::cout << "    [" << i << "] " << cfg.hosts[i].hostname
                              << " (slots=" << cfg.hosts[i].slots;
                    if (!cfg.hosts[i].gpus.empty()) {
                        std::cout << ", gpus=";
                        for (size_t j = 0; j < cfg.hosts[i].gpus.size(); ++j) {
                            if (j > 0)
                                std::cout << ",";
                            std::cout << cfg.hosts[i].gpus[j];
                        }
                    }
                    std::cout << ")\n";
                }
                if (cfg.ppn > 0) {
                    std::cout << "  PPN:           " << cfg.ppn << "\n";
                }
                if (!cfg.ssh_opts.empty()) {
                    std::cout << "  SSH Options:   " << cfg.ssh_opts << "\n";
                }
            } else {
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
            }
            std::cout << "  Ranks:         " << cfg.nranks << "\n"
                      << "  Application:   " << cfg.app_binary << "\n"
                      << "  Coord Dir:     " << cfg.coord_dir << "\n"
                      << "  Cleanup:       " << (cfg.cleanup ? "yes" : "no") << "\n";
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

        create_coord_dir(cfg.coord_dir);

        std::vector<pid_t> pids;
        pids.reserve(static_cast<size_t>(cfg.nranks));

        // Setup signal handler to cleanup children on interrupt
        g_child_pids = &pids;
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);

        if (cfg.use_ssh) {
            // Multi-node SSH mode
            int rank = 0;
            for (auto const& host : cfg.hosts) {
                int ranks_on_host = (cfg.ppn > 0) ? cfg.ppn : host.slots;

                for (int local_rank = 0; local_rank < ranks_on_host; ++local_rank) {
                    if (rank >= cfg.nranks)
                        break;

                    pid_t pid =
                        launch_rank_ssh(cfg, rank, host.hostname, local_rank, host.gpus);
                    pids.push_back(pid);

                    if (cfg.verbose) {
                        std::cout << "Launched rank " << rank << " (PID " << pid
                                  << ") on " << host.hostname;
                        if (!host.gpus.empty()) {
                            std::cout << " GPU "
                                      << host.gpus
                                             [static_cast<size_t>(local_rank)
                                              % host.gpus.size()];
                        }
                        std::cout << std::endl;
                    }
                    ++rank;
                }
            }
        } else {
            // Single-node local mode
            for (int rank = 0; rank < cfg.nranks; ++rank) {
                pid_t pid = launch_rank_local(cfg, rank);
                pids.push_back(pid);

                if (cfg.verbose) {
                    std::cout << "Launched rank " << rank << " (PID " << pid << ")";
                    if (!cfg.gpus.empty()) {
                        std::cout
                            << " on GPU "
                            << cfg.gpus[static_cast<size_t>(rank) % cfg.gpus.size()];
                    }
                    std::cout << std::endl;
                }
            }
        }

        if (cfg.verbose) {
            std::cout << "\nAll ranks launched. Waiting for completion...\n" << std::endl;
        }

        // Wait for all ranks to complete
        int exit_status = wait_for_ranks(pids);

        if (cfg.cleanup) {
            if (cfg.verbose) {
                std::cout << "Cleaning up coordination directory: " << cfg.coord_dir
                          << std::endl;
            }
            remove_dir_recursive(cfg.coord_dir);
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
