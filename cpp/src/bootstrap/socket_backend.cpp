/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <unistd.h>

#include <rapidsmpf/bootstrap/socket_backend.hpp>
#include <rapidsmpf/bootstrap/utils.hpp>

// NOTE: Do not use RAPIDSMPF_EXPECTS or RAPIDSMPF_FAIL in this file.
// Using these macros introduces a CUDA dependency via rapidsmpf/error.hpp.
// Prefer throwing standard exceptions instead.

namespace {

void write_all(int fd, void const* buf, std::size_t n) {
    auto const* ptr = static_cast<char const*>(buf);
    while (n > 0) {
        ssize_t w = ::write(fd, ptr, n);
        if (w < 0) {
            if (errno == EINTR)
                continue;
            throw std::runtime_error(
                "write() failed: " + std::string{std::strerror(errno)}
            );
        }
        ptr += w;
        n -= static_cast<std::size_t>(w);
    }
}

void read_all(int fd, void* buf, std::size_t n) {
    auto* ptr = static_cast<char*>(buf);
    while (n > 0) {
        ssize_t r = ::read(fd, ptr, n);
        if (r < 0) {
            if (errno == EINTR)
                continue;
            throw std::runtime_error(
                "read() failed: " + std::string{std::strerror(errno)}
            );
        }
        if (r == 0) {
            throw std::runtime_error("Connection closed unexpectedly");
        }
        ptr += r;
        n -= static_cast<std::size_t>(r);
    }
}

std::string read_line(int fd) {
    std::array<char, 512> buf{};
    std::size_t len = 0;
    for (;;) {
        char ch;
        ssize_t r = ::read(fd, &ch, 1);
        if (r < 0) {
            if (errno == EINTR)
                continue;
            throw std::runtime_error(
                "read() failed: " + std::string{std::strerror(errno)}
            );
        }
        if (r == 0) {
            throw std::runtime_error("Connection closed unexpectedly");
        }
        if (ch == '\n')
            break;
        if (len >= buf.size())
            throw std::runtime_error("Protocol line exceeds buffer size");
        buf[len++] = ch;
    }
    return {buf.data(), len};
}

std::string generate_token() {
    std::array<uint8_t, 32> bytes{};
    if (::getrandom(bytes.data(), bytes.size(), 0) != static_cast<ssize_t>(bytes.size()))
    {
        throw std::runtime_error(
            "getrandom() failed: " + std::string{std::strerror(errno)}
        );
    }
    static constexpr std::string_view hex = "0123456789abcdef";
    std::string result;
    result.reserve(bytes.size() * 2);
    for (uint8_t b : bytes) {
        result += hex[b >> 4];
        result += hex[b & 0xf];
    }
    return result;
}

}  // namespace

namespace rapidsmpf::bootstrap::detail {

struct SocketServer::State {
    std::mutex mu;

    // KV store: populated by rank 0's PUT commands
    std::map<std::string, std::string> kv;
    std::condition_variable kv_cv;

    // Barrier state: generation-based so multiple barriers can occur in sequence
    int barrier_gen{0};
    int barrier_count{0};
    std::condition_variable barrier_cv;

    // Sync state: same generation-based design as barrier
    int sync_gen{0};
    int sync_count{0};
    std::condition_variable sync_cv;

    int nranks;
    std::string token;  // 64-char hex
    std::atomic<bool> shutdown{false};
    std::atomic<int> authenticated_count{0};
    std::atomic<int> active_connections{0};
};

SocketServer::SocketServer(int nranks) : state_{std::make_shared<State>()} {
    state_->nranks = nranks;
    state_->token = generate_token();
    token_ = state_->token;

    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        throw std::runtime_error("socket() failed: " + std::string{std::strerror(errno)});
    }

    int opt = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = ::inet_addr("127.0.0.1");
    addr.sin_port = 0;  // let OS pick an ephemeral port

    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(listen_fd_);
        throw std::runtime_error("bind() failed: " + std::string{std::strerror(errno)});
    }

    if (::listen(listen_fd_, nranks + 4) < 0) {
        ::close(listen_fd_);
        throw std::runtime_error("listen() failed: " + std::string{std::strerror(errno)});
    }

    sockaddr_in bound{};
    socklen_t bound_len = sizeof(bound);
    if (::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&bound), &bound_len) < 0) {
        ::close(listen_fd_);
        throw std::runtime_error(
            "getsockname() failed: " + std::string{std::strerror(errno)}
        );
    }

    address_ = "127.0.0.1:" + std::to_string(::ntohs(bound.sin_port));

    // Wakeup pipe: write end signals accept_loop() to exit on shutdown.
    // O_CLOEXEC prevents child rank processes from inheriting these fds.
    if (::pipe(wakeup_pipe_.data()) < 0) {
        ::close(listen_fd_);
        throw std::runtime_error("pipe() failed: " + std::string{std::strerror(errno)});
    }
    ::fcntl(wakeup_pipe_[0], F_SETFD, FD_CLOEXEC);
    ::fcntl(wakeup_pipe_[1], F_SETFD, FD_CLOEXEC);

    accept_thread_ = std::thread([this]() { accept_loop(); });
}

SocketServer::~SocketServer() {
    state_->shutdown.store(true, std::memory_order_release);

    // Signal accept_loop() to exit by writing to the wakeup pipe.
    // Closing listen_fd_ alone is not guaranteed to interrupt a blocking
    // accept() call in another thread; the pipe write is reliable.
    if (wakeup_pipe_[1] >= 0) {
        char c = 0;
        std::ignore = ::write(wakeup_pipe_[1], &c, 1);
        ::close(wakeup_pipe_[1]);
        wakeup_pipe_[1] = -1;
    }

    if (listen_fd_ >= 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
    }

    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }

    if (wakeup_pipe_[0] >= 0) {
        ::close(wakeup_pipe_[0]);
        wakeup_pipe_[0] = -1;
    }

    std::lock_guard<std::mutex> lk(handler_mutex_);
    for (auto& t : handler_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
}

void SocketServer::accept_loop() {
    while (!state_->shutdown.load(std::memory_order_acquire)) {
        // Wait for either a new connection or the shutdown wakeup signal.
        std::array<pollfd, 2> fds{
            {{.fd = listen_fd_, .events = POLLIN, .revents = 0},
             {.fd = wakeup_pipe_[0], .events = POLLIN, .revents = 0}}
        };
        int ret = ::poll(fds.data(), 2, -1);
        if (ret < 0) {
            if (errno == EINTR)
                continue;
            break;
        }
        if (fds[1].revents & POLLIN) {
            break;  // wakeup signal from destructor
        }
        if (!(fds[0].revents & POLLIN)) {
            continue;
        }

        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd =
            ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
                continue;
            break;
        }

        // Reject connections beyond `2 * nranks` to prevent thread exhaustion
        // from a flood of unauthenticated connections. Increment first and
        // check afterwards to avoid a TOCTOU race between threads.
        int active =
            state_->active_connections.fetch_add(1, std::memory_order_relaxed) + 1;
        if (active > 2 * state_->nranks) {
            state_->active_connections.fetch_sub(1, std::memory_order_relaxed);
            ::close(client_fd);
            continue;
        }

        std::lock_guard<std::mutex> lk(handler_mutex_);
        handler_threads_.emplace_back([this, client_fd]() {
            handle_connection(client_fd);
            state_->active_connections.fetch_sub(1, std::memory_order_relaxed);
        });
    }
}

void SocketServer::handle_connection(int fd) {
    struct FdGuard {
        int fd;

        ~FdGuard() {
            ::close(fd);
        }
    } guard{fd};

    try {
        // Enforce a short timeout for AUTH so lingering connections don't block.
        struct timeval tv{.tv_sec = 2, .tv_usec = 0};
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        std::string auth_line = read_line(fd);

        int rank = -1, nranks = -1;
        std::array<char, 65> token_buf{};
        if (std::sscanf(
                auth_line.c_str(),
                "AUTH rank=%d nranks=%d token=%64s",
                &rank,
                &nranks,
                token_buf.data()
            )
            != 3)
        {
            write_all(fd, "ERROR invalid AUTH\n", 19);
            return;
        }

        {
            std::lock_guard<std::mutex> lk(state_->mu);
            if (std::string(token_buf.data()) != state_->token) {
                write_all(fd, "ERROR bad token\n", 16);
                return;
            }
            if (rank < 0 || rank >= state_->nranks || nranks != state_->nranks) {
                write_all(fd, "ERROR invalid rank or nranks\n", 29);
                return;
            }
        }

        // Remove timeout for normal operation.
        tv = {.tv_sec = 0, .tv_usec = 0};
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        write_all(fd, "OK\n", 3);

        // Once all ranks have authenticated there is no reason to keep the
        // listen socket open. Signal the accept loop to stop so no further
        // connections can be accepted.
        if (state_->authenticated_count.fetch_add(1, std::memory_order_acq_rel) + 1
            == state_->nranks)
        {
            char c = 0;
            std::ignore = ::write(wakeup_pipe_[1], &c, 1);
        }

        for (;;) {
            std::string line;
            try {
                line = read_line(fd);
            } catch (...) {
                break;  // client disconnected
            }
            if (line.empty())
                continue;

            if (line.starts_with("PUT ")) {
                // PUT key=<key> valuelen=<N>
                std::array<char, 256> key_buf{};
                std::size_t valuelen = 0;
                if (std::sscanf(
                        line.c_str(),
                        "PUT key=%255s valuelen=%zu",
                        key_buf.data(),
                        &valuelen
                    )
                    != 2)
                {
                    write_all(fd, "ERROR bad PUT\n", 14);
                    continue;
                }
                if (rank != 0) {
                    std::vector<char> discard(valuelen);
                    if (valuelen > 0)
                        read_all(fd, discard.data(), valuelen);
                    write_all(fd, "ERROR only rank 0 may PUT\n", 26);
                    continue;
                }
                std::string value(valuelen, '\0');
                if (valuelen > 0)
                    read_all(fd, value.data(), valuelen);
                {
                    std::lock_guard<std::mutex> lk(state_->mu);
                    state_->kv[std::string(key_buf.data())] = std::move(value);
                    state_->kv_cv.notify_all();
                }
                write_all(fd, "OK\n", 3);

            } else if (line.starts_with("GET ")) {
                // GET key=<key> timeout=<ms>
                std::array<char, 256> key_buf{};
                long long timeout_ms = 0;
                if (std::sscanf(
                        line.c_str(),
                        "GET key=%255s timeout=%lld",
                        key_buf.data(),
                        &timeout_ms
                    )
                    != 2)
                {
                    write_all(fd, "ERROR bad GET\n", 14);
                    continue;
                }
                std::string key{key_buf.data()};
                std::string value;
                bool found = false;
                {
                    std::unique_lock<std::mutex> lk(state_->mu);
                    auto deadline = std::chrono::steady_clock::now()
                                    + std::chrono::milliseconds(timeout_ms);
                    while (state_->kv.find(key) == state_->kv.end()) {
                        if (state_->kv_cv.wait_until(lk, deadline)
                            == std::cv_status::timeout)
                            break;
                    }
                    auto it = state_->kv.find(key);
                    if (it != state_->kv.end()) {
                        value = it->second;
                        found = true;
                    }
                }
                if (found) {
                    std::string header =
                        "VALUE valuelen=" + std::to_string(value.size()) + "\n";
                    write_all(fd, header.c_str(), header.size());
                    if (!value.empty())
                        write_all(fd, value.data(), value.size());
                } else {
                    write_all(fd, "TIMEOUT\n", 8);
                }

            } else if (line == "BARRIER") {
                std::unique_lock<std::mutex> lk(state_->mu);
                int gen = state_->barrier_gen;
                if (++state_->barrier_count == state_->nranks) {
                    ++state_->barrier_gen;
                    state_->barrier_count = 0;
                    state_->barrier_cv.notify_all();
                } else {
                    state_->barrier_cv.wait(lk, [&] {
                        return state_->barrier_gen != gen;
                    });
                }
                lk.unlock();
                write_all(fd, "OK\n", 3);

            } else if (line == "SYNC") {
                std::unique_lock<std::mutex> lk(state_->mu);
                int gen = state_->sync_gen;
                if (++state_->sync_count == state_->nranks) {
                    ++state_->sync_gen;
                    state_->sync_count = 0;
                    state_->sync_cv.notify_all();
                } else {
                    state_->sync_cv.wait(lk, [&] { return state_->sync_gen != gen; });
                }
                lk.unlock();
                write_all(fd, "OK\n", 3);

            } else {
                write_all(fd, "ERROR unknown command\n", 22);
            }
        }
    } catch (std::exception const&) {
        // Connection error or unexpected close, silently clean up.
    }
}

SocketBackend::SocketBackend(Context ctx) : ctx_{std::move(ctx)} {
    auto require_env = [](char const* name) -> std::string {
        char const* v = std::getenv(name);
        if (!v)
            throw std::runtime_error(
                std::string("SocketBackend: environment variable not set: ") + name
            );
        return v;
    };

    std::string addr = require_env("RRUN_SOCKET_ADDR");
    std::string token = require_env("RRUN_SOCKET_TOKEN");

    auto colon = addr.rfind(':');
    if (colon == std::string::npos) {
        throw std::runtime_error("RRUN_SOCKET_ADDR must be 'host:port', got: " + addr);
    }
    std::string host = addr.substr(0, colon);
    int port = std::stoi(addr.substr(colon + 1));

    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) {
        throw std::runtime_error("socket() failed: " + std::string{std::strerror(errno)});
    }

    sockaddr_in server{};
    server.sin_family = AF_INET;
    server.sin_port = htons(static_cast<uint16_t>(port));
    if (::inet_pton(AF_INET, host.c_str(), &server.sin_addr) != 1) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("inet_pton() failed for host: " + host);
    }

    if (::connect(fd_, reinterpret_cast<sockaddr const*>(&server), sizeof(server)) < 0) {
        int err = errno;
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error(
            "connect() to " + addr + " failed: " + std::string{std::strerror(err)}
        );
    }

    send_line(
        "AUTH rank=" + std::to_string(ctx_.rank)
        + " nranks=" + std::to_string(ctx_.nranks) + " token=" + token
    );

    std::string response = recv_line();
    if (response != "OK") {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("SocketBackend AUTH rejected: " + response);
    }
}

SocketBackend::~SocketBackend() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

void SocketBackend::send_line(std::string const& line) {
    std::string msg = line + "\n";
    write_all(fd_, msg.c_str(), msg.size());
}

std::string SocketBackend::recv_line() {
    return read_line(fd_);
}

void SocketBackend::send_bytes(void const* data, std::size_t n) {
    write_all(fd_, data, n);
}

void SocketBackend::recv_bytes(void* data, std::size_t n) {
    read_all(fd_, data, n);
}

void SocketBackend::put(std::string const& key, std::string_view value) {
    if (ctx_.rank != 0) {
        throw std::runtime_error(
            "put() can only be called by rank 0, but was called by rank "
            + std::to_string(ctx_.rank)
        );
    }
    validate_key(key);
    send_line("PUT key=" + key + " valuelen=" + std::to_string(value.size()));
    if (!value.empty()) {
        send_bytes(value.data(), value.size());
    }
    std::string resp = recv_line();
    if (resp != "OK") {
        throw std::runtime_error("SocketBackend put() failed: " + resp);
    }
}

std::string SocketBackend::get(std::string const& key, Duration timeout) {
    validate_key(key);
    auto timeout_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(timeout).count();
    send_line("GET key=" + key + " timeout=" + std::to_string(timeout_ms));

    std::string resp = recv_line();
    if (resp == "TIMEOUT") {
        throw std::runtime_error(
            "Key '" + key + "' not available within " + std::to_string(timeout.count())
            + "s timeout"
        );
    }
    if (!resp.starts_with("VALUE ")) {
        throw std::runtime_error("SocketBackend get() unexpected response: " + resp);
    }
    std::size_t valuelen = 0;
    if (std::sscanf(resp.c_str(), "VALUE valuelen=%zu", &valuelen) != 1) {
        throw std::runtime_error("SocketBackend get() malformed VALUE header: " + resp);
    }
    std::string value(valuelen, '\0');
    if (valuelen > 0)
        recv_bytes(value.data(), valuelen);
    return value;
}

void SocketBackend::barrier() {
    send_line("BARRIER");
    std::string resp = recv_line();
    if (resp != "OK") {
        throw std::runtime_error("SocketBackend barrier() failed: " + resp);
    }
}

void SocketBackend::sync() {
    send_line("SYNC");
    std::string resp = recv_line();
    if (resp != "OK") {
        throw std::runtime_error("SocketBackend sync() failed: " + resp);
    }
}

}  // namespace rapidsmpf::bootstrap::detail
