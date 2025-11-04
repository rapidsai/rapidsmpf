/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
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

#include <rapidsmpf/error.hpp>

namespace {

static std::mutex output_mutex;

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
 * @return Vector of GPU IDs.
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
            RAPIDSMPF_FAIL("Invalid GPU ID: " + item, std::runtime_error);
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
            RAPIDSMPF_EXPECTS(
                i + 1 < argc, "Missing argument for -n", std::runtime_error
            );
            cfg.nranks = std::stoi(argv[++i]);
            RAPIDSMPF_EXPECTS(
                cfg.nranks > 0,
                "Invalid number of ranks: " + std::to_string(cfg.nranks),
                std::runtime_error
            );
        } else if (arg == "-g") {
            RAPIDSMPF_EXPECTS(
                i + 1 < argc, "Missing argument for -g", std::runtime_error
            );
            cfg.gpus = parse_gpu_list(argv[++i]);
        } else if (arg == "--tag-output") {
            cfg.tag_output = true;
        } else if (arg == "-d") {
            RAPIDSMPF_EXPECTS(
                i + 1 < argc, "Missing argument for -d", std::runtime_error
            );
            cfg.coord_dir = argv[++i];
        } else if (arg == "-x" || arg == "--set-env") {
            RAPIDSMPF_EXPECTS(
                i + 1 < argc, "Missing argument for -x/--set-env", std::runtime_error
            );
            std::string env_spec = argv[++i];
            auto eq_pos = env_spec.find('=');
            RAPIDSMPF_EXPECTS(
                eq_pos != std::string::npos,
                "Invalid environment variable format: " + env_spec
                    + ". Expected VAR=value",
                std::runtime_error
            );
            std::string var_name = env_spec.substr(0, eq_pos);
            std::string var_value = env_spec.substr(eq_pos + 1);
            RAPIDSMPF_EXPECTS(
                !var_name.empty(), "Empty environment variable name", std::runtime_error
            );
            cfg.env_vars[var_name] = var_value;
        } else if (arg == "-v") {
            cfg.verbose = true;
        } else if (arg == "--no-cleanup") {
            cfg.cleanup = false;
        } else if (arg[0] == '-') {
            RAPIDSMPF_FAIL("Unknown option: " + arg, std::runtime_error);
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
    RAPIDSMPF_EXPECTS(
        !cfg.app_binary.empty(), "Missing application binary", std::runtime_error
    );

    // Single-node mode validation
    RAPIDSMPF_EXPECTS(
        cfg.nranks > 0,
        "Number of ranks (-n) must be specified and positive",
        std::runtime_error
    );

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
    RAPIDSMPF_EXPECTS(
        pipe(pipe_out) >= 0,
        "Failed to create stdout pipe: " + std::string{std::strerror(errno)},
        std::runtime_error
    );
    if (!combine_stderr) {
        if (pipe(pipe_err) < 0) {
            close(pipe_out[0]);
            close(pipe_out[1]);
            RAPIDSMPF_FAIL(
                "Failed to create stderr pipe: " + std::string{std::strerror(errno)},
                std::runtime_error
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
        RAPIDSMPF_FAIL(
            "Failed to fork: " + std::string{std::strerror(errno)}, std::runtime_error
        );
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
    return fork_with_piped_stdio(
        out_fd_stdout,
        out_fd_stderr,
        /*combine_stderr*/ false,
        [&cfg, rank]() {
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
            if (fd < 0)
                return;
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
                        if (!tag.empty())
                            fputs(tag.c_str(), out);
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
                std::cout << "Launched rank " << rank << " (PID " << pid << ")";
                if (!cfg.gpus.empty()) {
                    std::cout << " on GPU "
                              << cfg.gpus[static_cast<size_t>(rank) % cfg.gpus.size()];
                }
                std::cout << std::endl;
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
            if (th.joinable())
                th.join();
        }

        if (cfg.cleanup) {
            if (cfg.verbose) {
                std::cout << "Cleaning up coordination directory: " << cfg.coord_dir
                          << std::endl;
            }
            {
                std::error_code ec;
                std::filesystem::remove_all(cfg.coord_dir, ec);
                if (ec) {
                    std::cerr << "Warning: Failed to cleanup directory: " << cfg.coord_dir
                              << ": " << ec.message() << std::endl;
                }
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
