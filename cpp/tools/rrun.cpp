/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
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
 * @brief Configuration for the rrun launcher.
 */
struct Config {
    int nranks{1};  // Total number of ranks
    std::string app_binary;  // Application binary path
    std::vector<std::string> app_args;  // Arguments to pass to application
    std::vector<int> gpus;  // GPU IDs to use (empty = auto-detect)
    std::string coord_dir;  // Coordination directory
    bool verbose{false};  // Verbose output
    bool cleanup{true};  // Cleanup coordination directory on exit
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
    std::cout << "rrun - RapidsMPF Process Launcher\n\n"
              << "Usage: " << prog_name << " [options] <application> [app_args...]\n\n"
              << "Options:\n"
              << "  -n <nranks>        Number of ranks to launch (required)\n"
              << "  -g <gpu_list>      Comma-separated list of GPU IDs (e.g., 0,1,2,3)\n"
              << "                     If not specified, auto-detect available GPUs\n"
              << "  -d <coord_dir>     Coordination directory (default: "
                 "/tmp/rrun_<random>)\n"
              << "  -v                 Verbose output\n"
              << "  --no-cleanup       Don't cleanup coordination directory on exit\n"
              << "  -h, --help         Display this help message\n\n"
              << "Environment Variables:\n"
              << "  CUDA_VISIBLE_DEVICES is set for each rank based on GPU assignment\n\n"
              << "Examples:\n"
              << "  # Launch 2 ranks with auto-detected GPUs:\n"
              << "  rrun -n 2 ./bench_comm -C ucxx -O all-to-all\n\n"
              << "  # Launch 4 ranks on specific GPUs:\n"
              << "  rrun -n 4 -g 0,1,2,3 ./bench_comm -C ucxx\n\n"
              << "  # Launch with verbose output:\n"
              << "  rrun -v -n 2 ./my_app --arg1 value1\n"
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
        } else if (arg == "-d") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing argument for -d");
            }
            cfg.coord_dir = argv[++i];
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

    return cfg;
}

/**
 * @brief Launch a single rank.
 */
pid_t launch_rank(Config const& cfg, int rank) {
    pid_t pid = fork();

    if (pid < 0) {
        throw std::runtime_error("Failed to fork: " + std::string{std::strerror(errno)});
    } else if (pid == 0) {
        // Child process

        // Preserve parent's LD_LIBRARY_PATH (important for development builds)
        // No need to set it explicitly as it's inherited from parent

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
            std::cout << "rrun configuration:\n"
                      << "  Ranks:         " << cfg.nranks << "\n"
                      << "  Application:   " << cfg.app_binary << "\n"
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
            std::cout << "  Coord Dir:     " << cfg.coord_dir << "\n"
                      << "  Cleanup:       " << (cfg.cleanup ? "yes" : "no") << "\n"
                      << std::endl;
        }

        create_coord_dir(cfg.coord_dir);

        std::vector<pid_t> pids;
        pids.reserve(static_cast<size_t>(cfg.nranks));

        // Setup signal handler to cleanup children on interrupt
        g_child_pids = &pids;
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);

        // Launch all ranks
        for (int rank = 0; rank < cfg.nranks; ++rank) {
            pid_t pid = launch_rank(cfg, rank);
            pids.push_back(pid);

            if (cfg.verbose) {
                std::cout << "Launched rank " << rank << " (PID " << pid << ")";
                if (!cfg.gpus.empty()) {
                    std::cout << " on GPU "
                              << cfg.gpus[static_cast<size_t>(rank) % cfg.gpus.size()];
                }
                std::cout << std::endl;
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
