/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
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

class StreamingMemoryReserveOrWait : public BaseStreamingFixture {
  public:
    // Create a buffer resource that can be controlled by set_mem_avail()
    // and get_mem_avail().
    void SetUp() override {
        auto dev_mem_available = [this]() -> std::int64_t {
            return mem_avail_.load(std::memory_order_acquire);
        };
        SetUpWithThreads(1, {{MemoryType::DEVICE, dev_mem_available}});
    }

  protected:
    void set_mem_avail(std::int64_t size) {
        return mem_avail_.store(size, std::memory_order_release);
    }

    std::int64_t get_mem_avail() {
        return mem_avail_.load(std::memory_order_acquire);
    }

  private:
    std::atomic<std::int64_t> mem_avail_{0};
};

TEST_F(StreamingMemoryReserveOrWait, Timeout) {
    // Set no available memory so we expect to timeout.
    set_mem_avail(0);
    coro::sync_wait([](std::shared_ptr<Context> ctx) -> Node {
        MemoryReserveOrWait mrow{MemoryType::DEVICE, ctx, std::chrono::milliseconds{1}};
        auto res = co_await mrow.reserve_or_wait(10, 0);
        EXPECT_EQ(res.size(), 0);
    }(ctx));
}

TEST_F(StreamingMemoryReserveOrWait, ShutdownEarly) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    }
    MemoryReserveOrWait mrow{MemoryType::DEVICE, ctx, std::chrono::seconds{100}};

    // Create a reserve request while no memory is available.
    set_mem_avail(0);
    std::vector<Node> nodes;
    nodes.push_back([](MemoryReserveOrWait& mrow) -> Node {
        EXPECT_THROW(
            std::ignore = co_await mrow.reserve_or_wait(10, 1), std::runtime_error
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

TEST_F(StreamingMemoryReserveOrWait, CheckPriority) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    }
    ReserveLog log;
    MemoryReserveOrWait mrow{MemoryType::DEVICE, ctx, std::chrono::seconds{100}};

    // Create two reserve request while no memory is available.
    set_mem_avail(0);
    std::vector<Node> nodes;
    // One request with `future_release_potential = 1`.
    nodes.push_back([](ReserveLog& log, MemoryReserveOrWait& mrow) -> Node {
        auto res = co_await mrow.reserve_or_wait(10, 1);
        EXPECT_EQ(res.size(), 10);
        log.add(1, std::move(res));
    }(log, mrow));
    // And one request with `future_release_potential = 2`.
    nodes.push_back([](ReserveLog& log, MemoryReserveOrWait& mrow) -> Node {
        auto res = co_await mrow.reserve_or_wait(10, 2);
        EXPECT_EQ(res.size(), 10);
        log.add(2, std::move(res));
    }(log, mrow));

    // Run the pipeline on a dedicated thread.
    std::thread thd(run_streaming_pipeline, std::move(nodes));

    // We must ensure that both nodes have submitted their reservation requests
    // before checking the order in which MemoryReserveOrWait processed them.
    // First, we wait until both requests have been submitted (`mrow.size() == 2`),
    // then we ensure that periodic_memory_check() has had a chance to perform
    // at least one iteration.
    {
        std::uint64_t counter;
        while (mrow.size() < 2) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        counter = mrow.periodic_memory_check_counter();
        while (counter >= mrow.periodic_memory_check_counter()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // Make memory available, which should priorites the second node
    // with the highest "future_release_potential".
    set_mem_avail(20);
    thd.join();
    EXPECT_EQ(log.log.at(0).first, 2);
    EXPECT_EQ(log.log.at(1).first, 1);
}
