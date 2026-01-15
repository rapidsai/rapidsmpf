/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <thread>

#include <gtest/gtest.h>

#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/memory_reserve_or_wait.hpp>
#include <rapidsmpf/utils.hpp>

#include "base_streaming_fixture.hpp"

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;

struct ReserveOrWaitParam {
    int num_threads;
};

class StreamingMemoryReserveOrWait
    : public BaseStreamingFixture,
      public ::testing::WithParamInterface<ReserveOrWaitParam> {
  public:
    void SetUp() override {
        auto dev_mem_available = [this]() -> std::int64_t {
            return mem_avail_.load(std::memory_order_acquire);
        };
        SetUpWithThreads(
            GetParam().num_threads, {{rapidsmpf::MemoryType::DEVICE, dev_mem_available}}
        );
    }

  protected:
    void set_mem_avail(std::int64_t size) {
        mem_avail_.store(size, std::memory_order_release);
    }

    std::int64_t get_mem_avail() {
        return mem_avail_.load(std::memory_order_acquire);
    }

  private:
    std::atomic<std::int64_t> mem_avail_{0};
};

INSTANTIATE_TEST_SUITE_P(
    StreamingMemoryReserveOrWaitParams,
    StreamingMemoryReserveOrWait,
    ::testing::Values(
        ReserveOrWaitParam{1},
        ReserveOrWaitParam{2},
        ReserveOrWaitParam{5},
        ReserveOrWaitParam{8}
    ),
    [](testing::TestParamInfo<ReserveOrWaitParam> const& info) {
        return "T" + std::to_string(info.param.num_threads);
    }
);

TEST_P(StreamingMemoryReserveOrWait, AccessorsReturnExpectedValues) {
    constexpr std::int64_t timeout_ms = 12345;
    config::Options options{
        {{"memory_reserve_timeout_ms", config::OptionValue(std::to_string(timeout_ms))}}
    };

    MemoryReserveOrWait mrow{options, MemoryType::DEVICE, ctx->executor(), ctx->br()};

    // Executor and buffer resource should match the context.
    EXPECT_EQ(mrow.executor(), ctx->executor());
    EXPECT_EQ(mrow.br(), ctx->br());

    // Timeout should match the configured value.
    EXPECT_EQ(
        std::chrono::duration_cast<std::chrono::milliseconds>(mrow.timeout()).count(),
        timeout_ms
    );
}

TEST_P(StreamingMemoryReserveOrWait, ShutdownEarly) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    };
    MemoryReserveOrWait mrow{
        // Use a very high timeout to effectively disable timeout in this test.
        config::Options({{"memory_reserve_timeout_ms", config::OptionValue("100000")}}),
        MemoryType::DEVICE,
        ctx->executor(),
        ctx->br()
    };

    // Create a reserve request while no memory is available.
    set_mem_avail(0);
    std::vector<Node> nodes;
    nodes.push_back([](MemoryReserveOrWait& mrow) -> Node {
        EXPECT_THROW(
            std::ignore = co_await mrow.reserve_or_wait(10, 0), std::runtime_error
        );
    }(mrow));

    // Run the pipeline on a dedicated thread.
    std::thread thd(run_streaming_pipeline, std::move(nodes));

    // Wait until the node has submitted its request (`mrow.size() == 1`).
    while (mrow.size() < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // We expect shutdown to make `reserve_or_wait()` throw std::runtime_error.
    coro::sync_wait(mrow.shutdown());
    thd.join();
}

struct ReserveLog {
    void add(uint64_t uid, MemoryReservation&& res) {
        std::lock_guard<std::mutex> lock(mutex);
        log.emplace_back(uid, std::move(res));
    }

    std::mutex mutex;
    std::vector<std::pair<uint64_t, MemoryReservation>> log;
};

TEST_P(StreamingMemoryReserveOrWait, CheckPriority) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    }
    ReserveLog log;
    MemoryReserveOrWait mrow{
        // Use a very high timeout to effectively disable timeout in this test.
        config::Options({{"memory_reserve_timeout_ms", config::OptionValue("100000")}}),
        MemoryType::DEVICE,
        ctx->executor(),
        ctx->br()
    };

    // Create two reserve requests while no memory is available.
    set_mem_avail(0);
    std::vector<Node> nodes;
    // One request with `net_memory_delta = 1`.
    nodes.push_back([](ReserveLog& log, MemoryReserveOrWait& mrow) -> Node {
        auto res = co_await mrow.reserve_or_wait(10, 1);
        EXPECT_EQ(res.size(), 10);
        log.add(1, std::move(res));
    }(log, mrow));
    // And one request with `net_memory_delta = 2`.
    nodes.push_back([](ReserveLog& log, MemoryReserveOrWait& mrow) -> Node {
        auto res = co_await mrow.reserve_or_wait(10, 2);
        EXPECT_EQ(res.size(), 10);
        log.add(2, std::move(res));
    }(log, mrow));

    // Run the pipeline on a dedicated thread.
    std::thread thd(run_streaming_pipeline, std::move(nodes));

    // Ensure both requests are submitted and periodic_memory_check has run at least once.
    while (mrow.size() < 2) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    auto const counter = mrow.periodic_memory_check_counter();
    while (mrow.periodic_memory_check_counter() <= counter) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Only enough memory for ONE request, so completion order reflects selection order.
    set_mem_avail(10);

    // Wait until at least one reservation completes.
    while (log.log.size() < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Smaller `net_memory_delta` has higher priority, so request 1 should complete first.
    EXPECT_EQ(log.log.at(0).first, 1);

    // Now allow the second request to complete.
    set_mem_avail(20);
    thd.join();
    EXPECT_EQ(log.log.at(1).first, 2);
}

TEST_P(StreamingMemoryReserveOrWait, RestartPeriodicTask) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    }

    MemoryReserveOrWait mrow{
        // Use a very high timeout to effectively disable timeout in this test.
        config::Options({{"memory_reserve_timeout_ms", config::OptionValue("100000")}}),
        MemoryType::DEVICE,
        ctx->executor(),
        ctx->br()
    };

    // Round 1: create a request, then make memory available.
    set_mem_avail(0);
    std::vector<Node> nodes1;
    nodes1.push_back([](MemoryReserveOrWait& mrow) -> Node {
        auto res = co_await mrow.reserve_or_wait(10, 0);
        EXPECT_EQ(res.size(), 10);
    }(mrow));

    std::thread thd1(run_streaming_pipeline, std::move(nodes1));
    while (mrow.size() < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    set_mem_avail(20);
    thd1.join();

    // Wait until the periodic task has had time to observe "empty" and exit.
    // (We cannot observe task completion directly, but we can at least ensure
    // there are no pending requests.)
    while (mrow.size() != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Round 2: make memory unavailable again, submit a new request, then satisfy it.
    set_mem_avail(0);
    std::vector<Node> nodes2;
    nodes2.push_back([](MemoryReserveOrWait& mrow) -> Node {
        auto res = co_await mrow.reserve_or_wait(10, 0);
        EXPECT_EQ(res.size(), 10);
    }(mrow));

    std::thread thd2(run_streaming_pipeline, std::move(nodes2));
    while (mrow.size() < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    set_mem_avail(20);
    thd2.join();
}

TEST_P(StreamingMemoryReserveOrWait, NoDeadlockWhenSpawningWithStaleHandle) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    }

    MemoryReserveOrWait mrow{
        // Use a very high timeout to effectively disable timeout in this test.
        config::Options({{"memory_reserve_timeout_ms", config::OptionValue("100000")}}),
        MemoryType::DEVICE,
        ctx->executor(),
        ctx->br()
    };

    // Do multiple rounds to increase the chance we hit the "task exiting" window.
    for (int i = 0; i < 50; ++i) {
        set_mem_avail(0);
        std::vector<Node> nodes;
        nodes.push_back([](MemoryReserveOrWait& mrow) -> Node {
            auto res = co_await mrow.reserve_or_wait(10, 0);
            EXPECT_EQ(res.size(), 10);
        }(mrow));

        std::thread thd(run_streaming_pipeline, std::move(nodes));

        while (mrow.size() < 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        set_mem_avail(20);
        thd.join();
    }
}

TEST_P(StreamingMemoryReserveOrWait, OverbookOnTimeoutReportsOverbookingBytes) {
    // Start with no available memory so the request cannot be satisfied normally.
    set_mem_avail(0);

    coro::sync_wait([](std::shared_ptr<Context> ctx) -> Node {
        MemoryReserveOrWait mrow{
            // Use a very small timeout to trigger timeout immediately.
            config::Options({{"memory_reserve_timeout_ms", config::OptionValue("1")}}),
            MemoryType::DEVICE,
            ctx->executor(),
            ctx->br()
        };
        auto [res, overbooked_bytes] = co_await mrow.reserve_or_wait_or_overbook(10, 0);
        EXPECT_EQ(res.size(), 10);
        EXPECT_EQ(overbooked_bytes, 10);
    }(ctx));
}

TEST_P(StreamingMemoryReserveOrWait, FailOnTimeoutThrowsOverflowError) {
    // Start with no available memory so the request cannot be satisfied.
    set_mem_avail(0);

    coro::sync_wait([](std::shared_ptr<Context> ctx) -> Node {
        MemoryReserveOrWait mrow{
            // Use a very small timeout to trigger timeout immediately.
            config::Options({{"memory_reserve_timeout_ms", config::OptionValue("1")}}),
            MemoryType::DEVICE,
            ctx->executor(),
            ctx->br()
        };
        EXPECT_THROW(
            std::ignore = co_await mrow.reserve_or_wait_or_fail(10, 0),
            std::overflow_error
        );
    }(ctx));
}
