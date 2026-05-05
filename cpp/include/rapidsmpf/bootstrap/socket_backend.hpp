/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <rapidsmpf/bootstrap/backend.hpp>
#include <rapidsmpf/bootstrap/bootstrap.hpp>

// NOTE: Do not use RAPIDSMPF_EXPECTS or RAPIDSMPF_FAIL in this file.
// Using these macros introduces a CUDA dependency via rapidsmpf/error.hpp.
// Prefer throwing standard exceptions instead.

namespace rapidsmpf::bootstrap::detail {

/**
 * @brief In-process TCP coordination server, intended to run inside rrun.
 *
 * Binds to 127.0.0.1:0 (OS assigns an ephemeral port), generates a 256-bit
 * random authentication token via getrandom(2), and starts an accept thread.
 */
class SocketServer {
  public:
    /**
     * @brief Start the server.
     *
     * Binds to 127.0.0.1:0, generates a 256-bit token, and starts the accept
     * thread. The server is ready to accept connections before this constructor
     * returns.
     *
     * @param nranks Number of rank clients expected to connect.
     * @throws std::runtime_error on socket/bind/listen/getrandom failure.
     */
    explicit SocketServer(int nranks);

    /**
     * @brief Shut down the server and join all threads.
     *
     * Closes the listen socket (which unblocks accept()), then joins the accept
     * thread and all connection-handler threads.
     */
    ~SocketServer();

    /**
     * @brief Returns the server address as "127.0.0.1:<port>".
     *
     * @return Reference to the address string, valid for the lifetime of this object.
     */
    [[nodiscard]] std::string const& address() const noexcept {
        return address_;
    }

    /**
     * @brief Returns the 64-character hex-encoded 256-bit authentication token.
     *
     * @return Reference to the token string, valid for the lifetime of this object.
     */
    [[nodiscard]] std::string const& token() const noexcept {
        return token_;
    }

    SocketServer(SocketServer const&) = delete;
    SocketServer& operator=(SocketServer const&) = delete;
    SocketServer(SocketServer&&) = delete;
    SocketServer& operator=(SocketServer&&) = delete;

  private:
    struct State;
    std::shared_ptr<State> state_;
    int listen_fd_{-1};
    std::array<int, 2> wakeup_pipe_{-1, -1};
    std::string address_;
    std::string token_;
    std::thread accept_thread_;
    std::mutex handler_mutex_;
    std::vector<std::thread> handler_threads_;

    void accept_loop();
    void handle_connection(int client_fd);
};

/**
 * @brief Socket-based coordination backend (client side, runs in each rank).
 *
 * Connects to a SocketServer started by rrun and uses it for all KV and
 * barrier operations. Authentication uses a shared token passed via
 * `RRUN_SOCKET_TOKEN` environment variable, preventing unauthorized processes
 * from participating in coordination.
 *
 * Required environment variables (set by rrun):
 * - `RRUN_SOCKET_ADDR`: "host:port" of the coordinator server
 * - `RRUN_SOCKET_TOKEN`: 64-hex-char authentication token
 * - `RRUN_RANK`: This process's rank (0-indexed)
 * - `RRUN_NRANKS`: Total number of ranks
 */
class SocketBackend : public Backend {
  public:
    /** @copydoc FileBackend::FileBackend */
    explicit SocketBackend(Context ctx);

    ~SocketBackend() override;

    /** @copydoc Backend::put */
    void put(std::string const& key, std::string_view value) override;

    /** @copydoc Backend::get */
    std::string get(std::string const& key, Duration timeout) override;

    /** @copydoc Backend::barrier */
    void barrier() override;

    /** @copydoc Backend::sync */
    void sync() override;

  private:
    Context ctx_;
    int fd_{-1};

    void send_line(std::string const& line);
    std::string recv_line();
    void send_bytes(void const* data, std::size_t n);
    void recv_bytes(void* data, std::size_t n);
};

}  // namespace rapidsmpf::bootstrap::detail
