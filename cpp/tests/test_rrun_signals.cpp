/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief Tests for how rrun handles SIGINT, SIGTERM and its own death.
 *
 * These tests launch rrun themselves, so they run only when the test binary is
 * not started by rrun. The rrun binary is taken from the `RRUN_BINARY`
 * environment variable, or found on `PATH`.
 */

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <gtest/gtest.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <rapidsmpf/bootstrap/utils.hpp>

extern char** environ;

namespace {

using namespace std::chrono_literals;

constexpr auto timeout = 30s;

// Returns true once `cond` holds, or false if it does not within `timeout`.
bool wait_for(std::function<bool()> const& cond) {
    auto const deadline = std::chrono::steady_clock::now() + timeout;
    while (!cond()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(10ms);
    }
    return true;
}

// The rrun binary to test: `RRUN_BINARY` if set, otherwise `rrun` from `PATH`.
std::string rrun_binary() {
    char const* binary = std::getenv("RRUN_BINARY");
    return binary != nullptr ? binary : "rrun";
}

struct SpawnOptions {
    // If set, start in a new session with this terminal as stdin and controlling
    // terminal.
    std::string tty;
    // If set, write stderr to this file instead of discarding it.
    std::string stderr_path;
};

// Start `binary` with `args`, with its output discarded. Returns the pid, or -1
// if it could not be started, e.g. because `binary` was not found.
pid_t spawn(
    std::string const& binary,
    std::vector<std::string> const& args,
    SpawnOptions const& options = {}
) {
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(binary.c_str()));
    for (auto const& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);

    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    if (!options.tty.empty()) {
        posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETSID);
        posix_spawn_file_actions_addopen(
            &actions, STDIN_FILENO, options.tty.c_str(), O_RDWR, 0
        );
    }
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    if (options.stderr_path.empty()) {
        posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
    } else {
        posix_spawn_file_actions_addopen(
            &actions,
            STDERR_FILENO,
            options.stderr_path.c_str(),
            O_WRONLY | O_CREAT | O_TRUNC,
            0644
        );
    }
    pid_t pid = -1;
    int const err = ::posix_spawnp(&pid, argv[0], &actions, &attr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attr);
    return err == 0 ? pid : -1;
}

// Returns the child pids of `pid` whose command name is `comm`.
std::vector<pid_t> children(pid_t pid, std::string const& comm) {
    std::vector<pid_t> result;
    std::ifstream children_file(
        "/proc/" + std::to_string(pid) + "/task/" + std::to_string(pid) + "/children"
    );
    pid_t child = 0;
    while (children_file >> child) {
        std::ifstream comm_file("/proc/" + std::to_string(child) + "/comm");
        std::string name;
        if (comm_file >> name && name == comm) {
            result.push_back(child);
        }
    }
    return result;
}

// Returns true if `pid` has exited (gone or a zombie).
bool exited(pid_t pid) {
    std::ifstream stat_file("/proc/" + std::to_string(pid) + "/stat");
    std::string line;
    if (!std::getline(stat_file, line)) {
        return true;
    }
    // The state follows the command name, which is enclosed in parentheses.
    auto const pos = line.rfind(')');
    return pos != std::string::npos && pos + 2 < line.size() && line[pos + 2] == 'Z';
}

// Kill `pid` and reap it.
void kill_and_reap(pid_t pid) {
    ::kill(pid, SIGKILL);
    ::waitpid(pid, nullptr, 0);
}

// Expect rrun to exit with `expected` as its exit code.
void expect_exit_code(pid_t rrun, int expected) {
    int status = 0;
    if (!wait_for([&] { return ::waitpid(rrun, &status, WNOHANG) == rrun; })) {
        kill_and_reap(rrun);
        FAIL() << "rrun did not exit";
    }
    ASSERT_TRUE(WIFEXITED(status)) << "rrun did not exit normally";
    EXPECT_EQ(WEXITSTATUS(status), expected);
}

}  // namespace

class RrunSignalTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (rapidsmpf::bootstrap::is_running_with_rrun()) {
            GTEST_SKIP() << "These tests launch rrun and must not run under rrun";
        }
        binary_ = rrun_binary();
        pid_t const probe = spawn(binary_, {"--help"});
        if (probe < 0) {
            GTEST_SKIP() << "Cannot start " << binary_
                         << ", set RRUN_BINARY or put rrun on PATH";
        }
        ::waitpid(probe, nullptr, 0);
    }

    // Start rrun with `nranks` ranks running `command` and the rrun options
    // `options`, and wait until all ranks run the program named `comm`.
    // Returns -1 if they do not start.
    pid_t launch(
        int nranks,
        std::vector<std::string> const& command,
        std::string const& comm,
        std::vector<std::string> const& options = {},
        SpawnOptions const& spawn_options = {}
    ) {
        std::vector<std::string> args{"-n", std::to_string(nranks), "--bind-to", "none"};
        args.insert(args.end(), options.begin(), options.end());
        args.insert(args.end(), command.begin(), command.end());
        pid_t const pid = spawn(binary_, args, spawn_options);
        if (pid < 0) {
            return -1;
        }
        if (!wait_for([&] {
                return children(pid, comm).size() == static_cast<std::size_t>(nranks);
            }))
        {
            kill_and_reap(pid);
            return -1;
        }
        return pid;
    }

    std::string binary_;
};

// A forwarded SIGINT reaches the ranks, which exit from it. If rrun itself
// were killed by the signal, or the ranks had to be killed with SIGKILL, the
// exit code would differ. The long grace period rules out SIGKILL.
TEST_F(RrunSignalTest, SigintReachesRanks) {
    pid_t const rrun = launch(2, {"sleep", "60"}, "sleep", {"--kill-after", "3600"});
    ASSERT_GT(rrun, 0) << "ranks did not start";
    ::kill(rrun, SIGINT);
    expect_exit_code(rrun, 128 + SIGINT);
}

// A rank that ignores SIGINT is killed once the grace period has passed.
TEST_F(RrunSignalTest, RankIgnoringSignalIsKilled) {
    pid_t const rrun = launch(
        1, {"sh", "-c", "trap '' INT; exec sleep 60"}, "sleep", {"--kill-after", "1"}
    );
    ASSERT_GT(rrun, 0) << "ranks did not start";
    ::kill(rrun, SIGINT);
    expect_exit_code(rrun, 128 + SIGKILL);
}

// A second signal kills the remaining ranks without waiting for the grace
// period. SIGINT and then SIGTERM are used, because two pending SIGINTs merge
// into one.
TEST_F(RrunSignalTest, SecondSignalKillsRanks) {
    pid_t const rrun = launch(
        1,
        {"sh", "-c", "trap '' INT TERM; exec sleep 60"},
        "sleep",
        {"--kill-after", "3600"}
    );
    ASSERT_GT(rrun, 0) << "ranks did not start";
    ::kill(rrun, SIGINT);
    ::kill(rrun, SIGTERM);
    expect_exit_code(rrun, 128 + SIGKILL);
}

// Ranks are killed when rrun dies.
TEST_F(RrunSignalTest, RanksDieWithRrun) {
    pid_t const rrun = launch(2, {"sleep", "60"}, "sleep");
    ASSERT_GT(rrun, 0) << "ranks did not start";
    auto const ranks = children(rrun, "sleep");
    ASSERT_EQ(ranks.size(), 2);
    kill_and_reap(rrun);
    for (pid_t rank : ranks) {
        EXPECT_TRUE(wait_for([&] { return exited(rank); }))
            << "rank " << rank << " still running";
    }
}

// A Ctrl-C in a terminal reaches the ranks directly, so rrun does not forward
// it.
TEST_F(RrunSignalTest, TerminalSigintNotForwarded) {
    int const terminal = posix_openpt(O_RDWR | O_NOCTTY);
    if (terminal < 0 || grantpt(terminal) != 0 || unlockpt(terminal) != 0) {
        GTEST_SKIP() << "No pseudo-terminal available";
    }
    std::string const log = ::testing::TempDir() + "rrun_terminal_sigint.log";
    // The ranks ignore SIGINT, so they are still running when rrun handles it,
    // and rrun kills them right after.
    pid_t const rrun = launch(
        2,
        {"sh", "-c", "trap '' INT; exec sleep 60"},
        "sleep",
        {"--kill-after", "0"},
        {.tty = ptsname(terminal), .stderr_path = log}
    );
    if (rrun > 0) {
        // The terminal sends SIGINT to its foreground process group, which is
        // rrun and its ranks.
        EXPECT_EQ(::write(terminal, "\x03", 1), 1);
        expect_exit_code(rrun, 128 + SIGKILL);
    }
    ::close(terminal);
    ASSERT_GT(rrun, 0) << "ranks did not start";
    std::ifstream log_file(log);
    std::string const output{
        std::istreambuf_iterator<char>(log_file), std::istreambuf_iterator<char>()
    };
    EXPECT_NE(output.find("from the terminal"), std::string::npos) << output;
}

// A signal while no rank is running ends rrun, which first removes its
// coordination directory. The rank leaves a background `sleep` behind that
// holds its output pipe open, so rrun is still waiting for the rank's output
// after it has reaped the rank.
TEST_F(RrunSignalTest, SignalWithoutRanksEndsRrun) {
    std::string const dir = ::testing::TempDir() + "rrun_signal_coord_dir";
    std::string const pid_file = ::testing::TempDir() + "rrun_signal_sleep_pid";
    std::remove(pid_file.c_str());
    pid_t const rrun = spawn(
        binary_,
        {"-n",
         "1",
         "--bind-to",
         "none",
         "--file-backend",
         "-d",
         dir,
         "sh",
         "-c",
         "sleep 60 & echo $! > " + pid_file}
    );
    ASSERT_GT(rrun, 0) << "cannot start rrun";

    // Wait until the rank has started the `sleep` and rrun has reaped the rank.
    pid_t sleep_pid = 0;
    bool const reaped = wait_for([&] {
        std::ifstream pid_stream(pid_file);
        return (pid_stream >> sleep_pid) && children(rrun, "sh").empty();
    });
    bool const dir_created = std::filesystem::exists(dir);
    int status = 0;
    bool ended = false;
    if (reaped) {
        ::kill(rrun, SIGTERM);
        ended = wait_for([&] { return ::waitpid(rrun, &status, WNOHANG) == rrun; });
    }
    if (!ended) {
        kill_and_reap(rrun);
    }
    if (sleep_pid > 0) {
        ::kill(sleep_pid, SIGKILL);
    }
    ASSERT_TRUE(reaped) << "rrun did not reap the rank";
    ASSERT_TRUE(ended) << "rrun did not exit";
    EXPECT_TRUE(dir_created);
    EXPECT_TRUE(WIFSIGNALED(status) && WTERMSIG(status) == SIGTERM);
    EXPECT_FALSE(std::filesystem::exists(dir));
}
