/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rrun_utils.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <system_error>
#include <thread>

#include <signal.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cucascade/memory/topology_discovery.hpp>

#include <rrun/rrun.hpp>

#include <rapidsmpf/bootstrap/socket_backend.hpp>

// NOTE: Do not use RAPIDSMPF_EXPECTS or RAPIDSMPF_FAIL in this file.
// Using these macros introduces a CUDA dependency via rapidsmpf/error.hpp.
// Prefer throwing standard exceptions instead.

namespace rrun {

std::mutex output_mutex;

namespace {

// Signal mask rrun was started with, restored before exec'ing an application.
sigset_t original_signal_mask;

sigset_t forwarded_signals() {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGINT);
    sigaddset(&set, SIGTERM);
    return set;
}

char const* signal_name(int sig) {
    return sig == SIGINT ? "SIGINT" : "SIGTERM";
}

// Ranks that have been launched but not reaped yet, and how they were signalled.
// A pid is only reused after it has been reaped, so signalling the pids in
// `running` never hits another process.
struct Ranks {
    std::mutex mutex;
    std::set<pid_t> running;
    int forwarded_signal{0};  // Signal the ranks were sent, or 0.
    bool killed{false};  // The ranks were sent SIGKILL.
    std::string coord_dir;  // Removed if a signal ends rrun, or empty.
};

// Never destroyed, so the signal thread can use it until rrun exits.
Ranks& ranks() {
    static auto* const r = new Ranks();
    return *r;
}

// Set once a signal arrives, after which the output of the ranks is dropped.
std::atomic<bool> suppress_rank_output{false};

// Must be called with `r.mutex` held.
void signal_all(Ranks& r, int sig) {
    for (pid_t pid : r.running) {
        kill(pid, sig);
    }
}

// Must be called with `r.mutex` held.
void kill_all(Ranks& r) {
    if (!r.running.empty()) {
        std::cerr << "[rrun] Killing the remaining ranks." << std::endl;
        signal_all(r, SIGKILL);
    }
    r.killed = true;
}

// Removes the coordination directory and ends rrun by `sig`, as if it had not
// been blocked. Must be called with `r.mutex` held.
[[noreturn]] void exit_by_signal(Ranks& r, int sig) {
    if (!r.coord_dir.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(r.coord_dir, ec);
        if (ec) {
            std::cerr << "[rrun] Warning: Failed to cleanup directory: " << r.coord_dir
                      << ": " << ec.message() << std::endl;
        }
    }
    ::signal(sig, SIG_DFL);
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, sig);
    pthread_sigmask(SIG_UNBLOCK, &set, nullptr);
    raise(sig);
    _exit(128 + sig);
}

// Receives every SIGINT and SIGTERM sent to rrun. While no rank is running,
// a signal ends rrun. Otherwise the first signal is forwarded to the ranks,
// and the ranks still running `kill_after_seconds` later, or after another
// signal, are sent SIGKILL.
[[noreturn]] void handle_signals(int kill_after_seconds) {
    sigset_t const set = forwarded_signals();
    std::optional<std::chrono::steady_clock::time_point> kill_deadline;
    for (;;) {
        siginfo_t info{};
        int sig = 0;
        if (kill_deadline.has_value()) {
            auto const remaining = std::max(
                std::chrono::nanoseconds{
                    *kill_deadline - std::chrono::steady_clock::now()
                },
                std::chrono::nanoseconds{0}
            );
            auto const secs = std::chrono::duration_cast<std::chrono::seconds>(remaining);
            timespec const timeout{
                .tv_sec = static_cast<time_t>(secs.count()),
                .tv_nsec = static_cast<long>((remaining - secs).count())
            };
            sig = sigtimedwait(&set, &info, &timeout);
        } else {
            sig = sigwaitinfo(&set, &info);
        }

        Ranks& r = ranks();
        std::lock_guard<std::mutex> lock(r.mutex);
        if (sig < 0) {
            if (errno == EAGAIN) {  // The kill deadline has passed.
                kill_all(r);
                kill_deadline.reset();
            }
            continue;
        }
        if (r.running.empty()) {
            exit_by_signal(r, sig);
        }
        if (r.forwarded_signal == 0) {
            suppress_rank_output.store(true, std::memory_order_relaxed);
            r.forwarded_signal = sig;
            // A terminal sends Ctrl-C to its whole foreground process group,
            // which includes the ranks, so it is not forwarded.
            if (info.si_code == SI_KERNEL) {
                std::cerr << "[rrun] Received " << signal_name(sig)
                          << " from the terminal, which also sent it to the ranks. ";
            } else {
                std::cerr << "[rrun] Forwarding " << signal_name(sig)
                          << " to all ranks. ";
                signal_all(r, sig);
            }
            std::cerr << "Ranks still running in " << kill_after_seconds
                      << "s are killed. Press Ctrl-C again to kill them now."
                      << std::endl;
            kill_deadline = std::chrono::steady_clock::now()
                            + std::chrono::seconds{kill_after_seconds};
        } else if (!r.killed) {
            kill_all(r);
            kill_deadline.reset();
        }
    }
}

}  // namespace

void block_forwarded_signals() {
    sigset_t const set = forwarded_signals();
    pthread_sigmask(SIG_BLOCK, &set, &original_signal_mask);
}

void start_signal_thread(int kill_after_seconds) {
    std::thread(handle_signals, kill_after_seconds).detach();
}

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

std::vector<int> detect_gpus() {
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

    pid_t const parent_pid = getpid();
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
        // Child: get killed if rrun dies, so ranks are never left running
        // without their launcher. Exit if rrun already died before this call.
        prctl(PR_SET_PDEATHSIG, SIGKILL);
        if (getppid() != parent_pid) {
            _exit(1);
        }

        // Redirect stdout/stderr
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
        // exec_application calls execvp). We must catch any exception here
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

[[noreturn]] void exec_application(Config const& cfg) {
    std::vector<char*> exec_args;
    exec_args.push_back(const_cast<char*>(cfg.app_binary.c_str()));
    for (auto const& arg : cfg.app_args) {
        exec_args.push_back(const_cast<char*>(arg.c_str()));
    }
    exec_args.push_back(nullptr);

    // A signal mask survives exec, so without this the application would start
    // with SIGINT and SIGTERM blocked.
    pthread_sigmask(SIG_SETMASK, &original_signal_mask, nullptr);

    execvp(cfg.app_binary.c_str(), exec_args.data());

    // If we get here, execvp failed
    std::cerr << "[rrun] Failed to execute " << cfg.app_binary << ": "
              << std::strerror(errno) << std::endl;
    _exit(1);
}

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

int launch_ranks_fork_based(
    Config const& cfg, int rank_offset, int ranks_per_task, int total_ranks
) {
    std::vector<LaunchedProcess> processes;
    processes.reserve(static_cast<std::size_t>(ranks_per_task));

    // Helper to create a forwarder thread for a given fd (returns default thread
    // if fd < 0).
    auto make_forwarder = [&](int fd, int rank, bool to_stderr) -> std::thread {
        if (fd < 0) {
            return {};
        }
        return std::thread([fd, rank, to_stderr, &cfg]() {
            FILE* stream = fdopen(fd, "r");
            if (!stream) {
                close(fd);
                return;
            }
            std::string tag =
                cfg.tag_output ? ("[" + std::to_string(rank) + "] ") : std::string{};
            char buffer[4096];
            while (fgets(buffer, sizeof(buffer), stream) != nullptr) {
                if (suppress_rank_output.load(std::memory_order_relaxed)) {
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
        pid_t pid = 0;
        {
            // Hold the lock from fork to registration, so a signal reaches every
            // rank. A rank forked after a signal gets that signal right away.
            Ranks& r = ranks();
            std::lock_guard<std::mutex> lock(r.mutex);
            pid = launch_rank_local(
                cfg, global_rank, local_rank, total_ranks, &fd_out, &fd_err
            );
            r.running.insert(pid);
            if (r.killed) {
                kill(pid, SIGKILL);
            } else if (r.forwarded_signal != 0) {
                kill(pid, r.forwarded_signal);
            }
        }

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

    std::cout << "\n[rrun] All ranks launched. Waiting for completion...\n" << std::endl;

    // Wait for all processes
    int exit_status = 0;
    for (std::size_t i = 0; i < processes.size(); ++i) {
        LaunchedProcess& proc = processes[i];
        int status = 0;
        int global_rank = rank_offset + static_cast<int>(i);
        // Wait without reaping, drop the pid from `running`, then reap it, so the
        // signal thread never signals a reaped pid.
        siginfo_t info{};
        if (waitid(P_PID, static_cast<id_t>(proc.pid), &info, WEXITED | WNOWAIT) == 0) {
            Ranks& r = ranks();
            std::lock_guard<std::mutex> lock(r.mutex);
            r.running.erase(proc.pid);
        }
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

int setup_launch_and_cleanup(
    Config& cfg,
    int rank_offset,
    int ranks_per_task,
    int total_ranks,
    std::string const& coord_dir_hint
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
        if (cfg.cleanup) {
            Ranks& r = ranks();
            std::lock_guard<std::mutex> lock(r.mutex);
            r.coord_dir = cfg.coord_dir;
        }
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
        {
            Ranks& r = ranks();
            std::lock_guard<std::mutex> lock(r.mutex);
            r.coord_dir.clear();
        }
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

}  // namespace rrun
