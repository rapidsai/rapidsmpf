/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
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
 * The server accepts connections for its whole lifetime.
 *
 * Every authenticated connection belongs to a session. The k-th connection
 * that authenticates as rank `r` joins session k, and a session is complete
 * once all ranks have joined it. The KV store, barriers and syncs are scoped
 * to a session, so each bootstrap round sees only its own keys. A session is
 * discarded once it is complete and all its connections have closed.
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
     * Stops the accept thread and closes the listen socket. Then shuts down
     * all open client connections, wakes the handlers waiting in a barrier, a
     * sync or a get, and joins all connection-handler threads. Calls that
     * clients still have in flight fail.
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

    /**
     * @brief Returns the number of accepted client connections still open.
     *
     * The result is a snapshot. Connections still waiting in the listen
     * backlog are not counted.
     *
     * @return The number of open connections, authenticated or not.
     */
    [[nodiscard]] int num_open_connections() const;

    /**
     * @brief Returns the number of connection-handler threads not yet joined.
     *
     * The result is a snapshot.
     *
     * @return The number of handler threads.
     */
    [[nodiscard]] std::size_t num_handler_threads() const;

    /**
     * @brief Returns the number of connection handlers waiting for other ranks.
     *
     * Counts the handlers waiting in a barrier, a sync, or a get of a key that
     * is not set yet. The result is a snapshot.
     *
     * @return The number of waiting handlers.
     */
    [[nodiscard]] int num_waiting_handlers() const;

    SocketServer(SocketServer const&) = delete;
    SocketServer& operator=(SocketServer const&) = delete;
    SocketServer(SocketServer&&) = delete;
    SocketServer& operator=(SocketServer&&) = delete;

  private:
    struct State;

    /// @brief A connection-handler thread and a flag it sets when it is done.
    struct HandlerThread {
        std::thread thread;  ///< The handler thread.
        std::shared_ptr<std::atomic<bool>> done;  ///< Set when the handler returns.
    };

    std::shared_ptr<State> state_;
    int listen_fd_{-1};
    std::array<int, 2> wakeup_pipe_{-1, -1};
    std::string address_;
    std::string token_;
    std::thread accept_thread_;
    mutable std::mutex handler_mutex_;
    std::vector<HandlerThread> handler_threads_;

    void accept_loop();
    void reap_finished_handlers();
    void handle_connection(int client_fd);
};

/// @brief Default timeout for the coordination server to answer `AUTH`.
inline constexpr Duration default_auth_timeout = std::chrono::seconds{30};

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
 *
 * A process may create any number of backends one after another, each one
 * joining the next session on the server. At most one backend per rank may
 * be alive in a process at a time.
 */
class SocketBackend : public Backend {
  public:
    /**
     * @brief Connect and authenticate to the coordination server.
     *
     * @param ctx Bootstrap context containing rank and nranks.
     * @param auth_timeout How long to wait for the server to answer `AUTH`.
     * @throws std::runtime_error if a backend for `ctx.rank` is already alive in
     * this process, if the connection fails, or if the server rejects or does not
     * answer `AUTH`.
     */
    explicit SocketBackend(Context ctx, Duration auth_timeout = default_auth_timeout);

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
    /// @brief Registers a rank as having a live backend in this process.
    class RankRegistration {
      public:
        explicit RankRegistration(Rank rank);
        ~RankRegistration();
        RankRegistration(RankRegistration const&) = delete;
        RankRegistration& operator=(RankRegistration const&) = delete;
        RankRegistration(RankRegistration&&) = delete;
        RankRegistration& operator=(RankRegistration&&) = delete;

      private:
        Rank rank_;
    };

    RankRegistration registration_;  // must be initialized before ctx_
    Context ctx_;
    int fd_{-1};

    void send_line(std::string const& line);
    std::string recv_line();
    void send_bytes(void const* data, std::size_t n);
    void recv_bytes(void* data, std::size_t n);
};

}  // namespace rapidsmpf::bootstrap::detail
