/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <latch>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <rapidsmpf/bootstrap/backend.hpp>
#include <rapidsmpf/bootstrap/bootstrap.hpp>
#include <rapidsmpf/bootstrap/socket_backend.hpp>

namespace {

using rapidsmpf::bootstrap::BackendType;
using rapidsmpf::bootstrap::Context;
using rapidsmpf::bootstrap::detail::SocketBackend;
using rapidsmpf::bootstrap::detail::SocketServer;

// Sets the environment variables a SocketBackend reads, for the scope of the object.
class SocketEnv {
  public:
    SocketEnv(std::string const& addr, std::string const& token) {
        ::setenv("RRUN_SOCKET_ADDR", addr.c_str(), 1);
        ::setenv("RRUN_SOCKET_TOKEN", token.c_str(), 1);
    }

    explicit SocketEnv(SocketServer const& server)
        : SocketEnv(server.address(), server.token()) {}

    ~SocketEnv() {
        ::unsetenv("RRUN_SOCKET_ADDR");
        ::unsetenv("RRUN_SOCKET_TOKEN");
    }

    SocketEnv(SocketEnv const&) = delete;
    SocketEnv& operator=(SocketEnv const&) = delete;
    SocketEnv(SocketEnv&&) = delete;
    SocketEnv& operator=(SocketEnv&&) = delete;
};

Context make_ctx(int rank, int nranks) {
    return Context{
        .rank = rank,
        .nranks = nranks,
        .type = BackendType::SOCKET,
    };
}

// Run one thread per rank against `server`, each calling body(rank).
// Joins all threads, then re-throws the first captured exception (if any).
void run_threads(int nranks, std::function<void(int)> const& body) {
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(nranks));

    for (int rank = 0; rank < nranks; ++rank) {
        threads.emplace_back([rank, &body, &errors]() {
            try {
                body(rank);
            } catch (...) {
                errors[static_cast<std::size_t>(rank)] = std::current_exception();
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    for (auto const& e : errors) {
        if (e) {
            std::rethrow_exception(e);  // NOLINT(hicpp-exception-baseclass)
        }
    }
}

// Run one bootstrap round on `server`: N client threads each create a
// SocketBackend and run body(backend, rank).
void run_with_ranks(
    SocketServer& server, int nranks, std::function<void(SocketBackend&, int)> body
) {
    SocketEnv env{server};
    run_threads(nranks, [&](int rank) {
        SocketBackend backend(make_ctx(rank, nranks));
        body(backend, rank);
    });
}

// Spin up a SocketServer and run one bootstrap round on it.
void run_with_ranks(int nranks, std::function<void(SocketBackend&, int)> body) {
    SocketServer server(nranks);
    run_with_ranks(server, nranks, std::move(body));
}

// Open a raw (unauthenticated) TCP connection to the server.
int connect_raw(SocketServer const& server) {
    auto const& addr = server.address();
    int port = std::stoi(addr.substr(addr.rfind(':') + 1));

    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
        return -1;

    sockaddr_in sin{};
    sin.sin_family = AF_INET;
    sin.sin_port = ::htons(static_cast<uint16_t>(port));
    ::inet_pton(AF_INET, "127.0.0.1", &sin.sin_addr);

    if (::connect(fd, reinterpret_cast<sockaddr const*>(&sin), sizeof(sin)) < 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

// Send a line (appending '\n') to fd and return the first response line.
std::string send_recv_line(int fd, std::string const& line) {
    std::string msg = line + "\n";
    auto bytes_left = msg.size();
    auto const* ptr = msg.data();
    while (bytes_left > 0) {
        auto written = ::write(fd, ptr, bytes_left);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(
                "write() failed: " + std::string{std::strerror(errno)}
            );
        }
        bytes_left -= static_cast<std::size_t>(written);
        ptr += written;
    }

    std::string response;
    char ch;
    while (::read(fd, &ch, 1) > 0) {
        if (ch == '\n')
            break;
        response += ch;
    }
    return response;
}

// Wait until the server has no open connection. A closed connection keeps its
// slot until its handler sees EOF. Returns false if that does not happen
// within `timeout`.
[[nodiscard]] bool wait_until_idle(
    SocketServer const& server,
    std::chrono::milliseconds timeout = std::chrono::seconds{10}
) {
    auto const deadline = std::chrono::steady_clock::now() + timeout;
    while (server.num_open_connections() > 0) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    return true;
}

// Authenticate a raw connection as `rank` and return the server's response.
std::string auth_raw(SocketServer const& server, int fd, int rank, int nranks) {
    return send_recv_line(
        fd,
        "AUTH rank=" + std::to_string(rank) + " nranks=" + std::to_string(nranks)
            + " token=" + server.token()
    );
}

// Returns true if the peer has closed `fd` (read returns EOF).
bool peer_closed(int fd) {
    char ch;
    for (;;) {
        auto r = ::read(fd, &ch, 1);
        if (r < 0 && errno == EINTR) {
            continue;
        }
        return r == 0;
    }
}

// Returns the message of the std::runtime_error thrown by `fn`, or an empty
// string if it throws nothing.
std::string error_message(std::function<void()> const& fn) {
    try {
        fn();
    } catch (std::runtime_error const& e) {
        return e.what();
    }
    return {};
}

}  // namespace

TEST(SocketServerTest, AuthAndConnect) {
    run_with_ranks(4, [](SocketBackend&, int) {});
}

TEST(SocketServerTest, PutGet) {
    run_with_ranks(4, [](SocketBackend& backend, int rank) {
        if (rank == 0) {
            backend.put("hello", "world ! ! !");
        }
        backend.sync();
        EXPECT_EQ(backend.get("hello", std::chrono::seconds{5}), "world ! ! !");
    });
}

TEST(SocketServerTest, Barrier) {
    std::atomic<int> counter{0};
    run_with_ranks(4, [&counter](SocketBackend& backend, int) {
        counter.fetch_add(1, std::memory_order_relaxed);
        backend.barrier();
        EXPECT_EQ(counter.load(std::memory_order_relaxed), 4);
    });
}

TEST(SocketServerTest, MultipleBarriers) {
    run_with_ranks(4, [](SocketBackend& backend, int rank) {
        for (int i = 0; i < 3; ++i) {
            std::string key = "iter" + std::to_string(i);
            std::string expected = std::to_string(i);
            if (rank == 0) {
                backend.put(key, expected);
            }
            backend.sync();
            EXPECT_EQ(backend.get(key, std::chrono::seconds{5}), expected);
            backend.barrier();
        }
    });
}

TEST(SocketServerTest, GetTimeout) {
    run_with_ranks(1, [](SocketBackend& backend, int) {
        EXPECT_THROW(
            backend.get("nonexistent", std::chrono::milliseconds{100}), std::runtime_error
        );
    });
}

TEST(SocketServerTest, PutNonRank0Throws) {
    run_with_ranks(2, [](SocketBackend& backend, int rank) {
        if (rank == 1) {
            EXPECT_THROW(backend.put("key", "value"), std::runtime_error);
        }
        backend.barrier();
    });
}

TEST(SocketServerTest, BinaryValue) {
    std::string binary(256, '\0');
    for (std::size_t i = 0; i < 256; ++i) {
        binary[i] = static_cast<char>(i);
    }
    run_with_ranks(2, [&binary](SocketBackend& backend, int rank) {
        if (rank == 0) {
            backend.put("blob", binary);
        }
        backend.sync();
        EXPECT_EQ(backend.get("blob", std::chrono::seconds{5}), binary);
    });
}

TEST(SocketServerTest, UnauthenticatedCommandRejected) {
    SocketServer server(1);
    int fd = connect_raw(server);
    ASSERT_GE(fd, 0);

    // Send a PUT without AUTH, server expects AUTH as the first line.
    std::string resp = send_recv_line(fd, "PUT key=foo valuelen=5");
    EXPECT_TRUE(resp.starts_with("ERROR")) << "Expected ERROR, got: " << resp;
    ::close(fd);
}

TEST(SocketServerTest, WrongTokenRejected) {
    SocketServer server(1);
    int fd = connect_raw(server);
    ASSERT_GE(fd, 0);

    std::string resp =
        send_recv_line(fd, "AUTH rank=0 nranks=1 token=" + std::string(64, 'x'));
    EXPECT_TRUE(resp.starts_with("ERROR")) << "Expected ERROR, got: " << resp;
    ::close(fd);
}

TEST(SocketServerTest, SequentialSessions) {
    SocketServer server(3);
    for (int round = 0; round < 3; ++round) {
        run_with_ranks(server, 3, [](SocketBackend& backend, int rank) {
            if (rank == 0) {
                backend.put("key", "value");
            }
            backend.sync();
            EXPECT_EQ(backend.get("key", std::chrono::seconds{5}), "value");
            backend.barrier();
        });
    }
}

TEST(SocketServerTest, StaleKeyNotVisible) {
    SocketServer server(2);
    run_with_ranks(server, 2, [](SocketBackend& backend, int rank) {
        if (rank == 0) {
            backend.put("k", "round0");
        }
        backend.sync();
    });
    run_with_ranks(server, 2, [](SocketBackend& backend, int) {
        EXPECT_THROW(
            backend.get("k", std::chrono::milliseconds{100}), std::runtime_error
        );
        backend.barrier();
    });
}

TEST(SocketServerTest, OverlappingSessions) {
    // Rank 0 leaves round 0 and enters round 1 while rank 1 is still in round 0.
    // Rank 1 must still see round 0's value.
    SocketServer server(2);
    SocketEnv env{server};
    std::latch rank0_in_round1{1};

    run_threads(2, [&](int rank) {
        if (rank == 0) {
            {
                SocketBackend backend(make_ctx(0, 2));
                backend.put("k", "round0");
                backend.sync();
            }
            SocketBackend backend(make_ctx(0, 2));
            backend.put("k", "round1");
            rank0_in_round1.count_down();
            backend.barrier();
            EXPECT_EQ(backend.get("k", std::chrono::seconds{5}), "round1");
        } else {
            {
                SocketBackend backend(make_ctx(1, 2));
                backend.sync();
                rank0_in_round1.wait();
                EXPECT_EQ(backend.get("k", std::chrono::seconds{5}), "round0");
            }
            SocketBackend backend(make_ctx(1, 2));
            EXPECT_EQ(backend.get("k", std::chrono::seconds{5}), "round1");
            backend.barrier();
        }
    });
}

TEST(SocketServerTest, HandlerThreadsReaped) {
    SocketServer server(1);
    for (int round = 0; round < 5; ++round) {
        run_with_ranks(server, 1, [](SocketBackend&, int) {});
        ASSERT_TRUE(wait_until_idle(server))
            << server.num_open_connections() << " connections still open";
    }
    // Handlers are reaped when the next connection is accepted, so only the
    // last one can be left.
    EXPECT_LE(server.num_handler_threads(), 1);
}

TEST(SocketServerTest, ConnectionCapStillEnforced) {
    SocketServer server(1);  // cap is 2 * nranks = 2
    int fd0 = connect_raw(server);
    ASSERT_GE(fd0, 0);
    EXPECT_EQ(auth_raw(server, fd0, 0, 1), "OK");
    int fd1 = connect_raw(server);
    ASSERT_GE(fd1, 0);
    EXPECT_EQ(auth_raw(server, fd1, 0, 1), "OK");

    // A third connection is closed by the server without a response.
    int fd2 = connect_raw(server);
    ASSERT_GE(fd2, 0);
    EXPECT_TRUE(peer_closed(fd2));
    ::close(fd2);

    // Once the open connections are closed, new ones are accepted again.
    ::close(fd0);
    ::close(fd1);
    ASSERT_TRUE(wait_until_idle(server))
        << server.num_open_connections() << " connections still open";
    int fd3 = connect_raw(server);
    ASSERT_GE(fd3, 0);
    EXPECT_EQ(auth_raw(server, fd3, 0, 1), "OK");
    ::close(fd3);
}

TEST(SocketServerTest, AuthReplyTimeout) {
    // A listening socket that never accepts: connect() completes through the
    // backlog, but AUTH is never answered.
    int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_GE(listen_fd, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = 0;
    ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    ASSERT_EQ(::bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    ASSERT_EQ(::listen(listen_fd, 1), 0);
    socklen_t len = sizeof(addr);
    ASSERT_EQ(::getsockname(listen_fd, reinterpret_cast<sockaddr*>(&addr), &len), 0);

    SocketEnv env{
        "127.0.0.1:" + std::to_string(::ntohs(addr.sin_port)), std::string(64, 'a')
    };
    auto msg = error_message([] {
        SocketBackend backend(make_ctx(0, 1), std::chrono::milliseconds{100});
    });
    EXPECT_NE(msg.find("did not answer AUTH"), std::string::npos) << msg;
    ::close(listen_fd);
}

TEST(SocketServerTest, ConcurrentBootstrapRejected) {
    SocketServer server(1);
    SocketEnv env{server};
    {
        SocketBackend first(make_ctx(0, 1));
        auto msg = error_message([] { SocketBackend second(make_ctx(0, 1)); });
        EXPECT_NE(msg.find("already bootstrapping"), std::string::npos) << msg;
        first.put("k", "first");
        first.sync();
        EXPECT_EQ(first.get("k", std::chrono::seconds{5}), "first");
    }
}

TEST(SocketServerTest, DestructorUnblocksWaitingRank) {
    // Rank 0 waits in a barrier that rank 1, which has already left, never
    // joins. Destroying the server must not hang, and rank 0's barrier fails.
    auto server = std::make_unique<SocketServer>(2);
    SocketEnv env{*server};
    std::optional<SocketBackend> rank1{make_ctx(1, 2)};
    SocketBackend rank0{make_ctx(0, 2)};
    rank1.reset();

    std::thread waiter([&rank0] { EXPECT_THROW(rank0.barrier(), std::runtime_error); });

    // Destroy the server on another thread, so that a hang fails the test
    // instead of blocking it.
    std::promise<void> destroyed;
    auto destroyed_future = destroyed.get_future();
    std::thread([&server, &destroyed] {
        server.reset();
        destroyed.set_value();
    }).detach();
    if (destroyed_future.wait_for(std::chrono::seconds{10}) != std::future_status::ready)
    {
        std::cerr << "~SocketServer did not return within 10s" << std::endl;
        std::_Exit(EXIT_FAILURE);
    }

    waiter.join();
}
