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

// Spin up a SocketServer and N client threads, run body(backend, rank) in each.
// Joins all threads, then re-throws the first captured exception (if any).
void run_with_ranks(int nranks, std::function<void(SocketBackend&, int)> body) {
    SocketServer server(nranks);
    ::setenv("RRUN_SOCKET_ADDR", server.address().c_str(), 1);
    ::setenv("RRUN_SOCKET_TOKEN", server.token().c_str(), 1);

    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(nranks));

    for (int rank = 0; rank < nranks; ++rank) {
        threads.emplace_back([rank, nranks, &body, &errors]() {
            try {
                Context ctx{
                    .rank = rank,
                    .nranks = nranks,
                    .type = BackendType::SOCKET,
                };
                SocketBackend backend(std::move(ctx));
                body(backend, rank);
            } catch (...) {
                errors[static_cast<std::size_t>(rank)] = std::current_exception();
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
    ::unsetenv("RRUN_SOCKET_ADDR");
    ::unsetenv("RRUN_SOCKET_TOKEN");

    for (auto const& e : errors) {
        if (e) {
            std::rethrow_exception(e);  // NOLINT(hicpp-exception-baseclass)
        }
    }
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

TEST(SocketServerTest, DestructorUnblocksWaitingRank) {
    // Rank 0 waits in a barrier that rank 1, which has already left, never
    // joins. Destroying the server must not hang, and rank 0's barrier fails.
    auto server = std::make_unique<SocketServer>(2);
    ::setenv("RRUN_SOCKET_ADDR", server->address().c_str(), 1);
    ::setenv("RRUN_SOCKET_TOKEN", server->token().c_str(), 1);
    auto make_ctx = [](int rank) {
        return Context{.rank = rank, .nranks = 2, .type = BackendType::SOCKET};
    };
    std::optional<SocketBackend> rank1{make_ctx(1)};
    SocketBackend rank0{make_ctx(0)};
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
    ::unsetenv("RRUN_SOCKET_ADDR");
    ::unsetenv("RRUN_SOCKET_TOKEN");
}
