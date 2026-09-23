/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/memory_reserve_or_wait.hpp>
#include <rapidsmpf/utils/misc.hpp>
#include <rapidsmpf/utils/string.hpp>

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
        // Drive device memory availability by mutating the DEVICE limit. No real
        // device allocations occur in these tests, so `memory_available(DEVICE)`
        // equals whatever limit we set.
        SetUpWithThreads(GetParam().num_threads, {{rapidsmpf::MemoryType::DEVICE, 0}});
    }

  protected:
    void set_mem_avail(std::int64_t size) {
        br->set_memory_limit(rapidsmpf::MemoryType::DEVICE, size);
    }

    std::int64_t get_mem_avail() {
        return br->memory_available(rapidsmpf::MemoryType::DEVICE);
    }

    // Buffer resource with statistics enabled and no periodic spill thread, so the
    // recorded stats come only from the reservation path under test. The fixture's
    // own `br` uses `Statistics::disabled()`.
    struct StatsBufferResource {
        std::shared_ptr<BufferResource> br;
        std::shared_ptr<Statistics> stats;
    };

    StatsBufferResource make_br_with_stats(std::int64_t device_limit) {
        auto stats = Statistics::create();
        auto br_with_stats = BufferResource::create(
            mr_cuda,
            rapidsmpf::PinnedMemoryDisabled,
            {{MemoryType::DEVICE, device_limit}},
            /* periodic_spill_check = */ std::nullopt,
            std::make_shared<StreamPool>(16),
            stats
        );
        return {.br = std::move(br_with_stats), .stats = std::move(stats)};
    }

    // Buffer resource with no periodic spill thread. The admission loop is then the
    // only thing that could spill, which is what the contract forbids.
    std::shared_ptr<BufferResource> make_br_without_periodic_spill(std::int64_t limit) {
        return BufferResource::create(
            mr_cuda,
            rapidsmpf::PinnedMemoryDisabled,
            {{rapidsmpf::MemoryType::DEVICE, limit}},
            /* periodic_spill_check = */ std::nullopt
        );
    }

    // Actor that reserves `size` bytes and expects a reservation of `expected` bytes.
    static Actor waiter(
        MemoryReserveOrWait& mrow,
        std::size_t size,
        std::int64_t net_memory_delta,
        std::size_t expected
    ) {
        auto res = co_await mrow.reserve_or_wait(size, net_memory_delta);
        EXPECT_EQ(res.size(), expected);
    }
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
    config::Options options{
        {{"memory_reserve_timeout", config::OptionValue("12345 ms")}}
    };

    MemoryReserveOrWait mrow{options, MemoryType::DEVICE, ctx->executor(), ctx->br()};

    // Executor and buffer resource should match the context.
    EXPECT_EQ(mrow.executor(), ctx->executor());
    EXPECT_EQ(mrow.br(), ctx->br());

    // Timeout should match the configured value.
    EXPECT_EQ(mrow.timeout(), parse_duration("12345 ms"));
}

TEST_P(StreamingMemoryReserveOrWait, ShutdownEarly) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    };
    MemoryReserveOrWait mrow{
        // Use a very high timeout to effectively disable timeout in this test.
        config::Options({{"memory_reserve_timeout", config::OptionValue("1 min")}}),
        MemoryType::DEVICE,
        ctx->executor(),
        ctx->br()
    };

    // Create a reserve request while no memory is available.
    set_mem_avail(0);
    std::vector<Actor> actors;
    actors.push_back([](MemoryReserveOrWait& mrow) -> Actor {
        EXPECT_THROW(
            std::ignore = co_await mrow.reserve_or_wait(10, 0), std::runtime_error
        );
    }(mrow));

    // Run the pipeline on a dedicated thread.
    std::thread thd(run_actor_network, std::move(actors));

    // Wait until the actor has submitted its request (`mrow.size() == 1`).
    while (mrow.size() < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // We expect shutdown to make `reserve_or_wait()` throw std::runtime_error.
    coro::sync_wait(mrow.shutdown());
    thd.join();
}

struct ReserveLog {
    void add(std::uint64_t uid, MemoryReservation&& res) {
        std::lock_guard<std::mutex> lock(mutex);
        log.emplace_back(uid, std::move(res));
    }

    std::size_t size() {
        std::lock_guard<std::mutex> lock(mutex);
        return log.size();
    }

    std::mutex mutex;
    std::vector<std::pair<std::uint64_t, MemoryReservation>> log;
};

TEST_P(StreamingMemoryReserveOrWait, CheckPriority) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    }
    ReserveLog log;
    MemoryReserveOrWait mrow{
        // Use a very high timeout to effectively disable timeout in this test.
        config::Options({{"memory_reserve_timeout", config::OptionValue("1 min")}}),
        MemoryType::DEVICE,
        ctx->executor(),
        ctx->br()
    };

    // Create two reserve requests while no memory is available.
    set_mem_avail(0);
    std::vector<Actor> actors;
    // One request with `net_memory_delta = 1`.
    actors.push_back([](ReserveLog& log, MemoryReserveOrWait& mrow) -> Actor {
        auto res = co_await mrow.reserve_or_wait(10, 1);
        EXPECT_EQ(res.size(), 10);
        log.add(1, std::move(res));
    }(log, mrow));
    // And one request with `net_memory_delta = 2`.
    actors.push_back([](ReserveLog& log, MemoryReserveOrWait& mrow) -> Actor {
        auto res = co_await mrow.reserve_or_wait(10, 2);
        EXPECT_EQ(res.size(), 10);
        log.add(2, std::move(res));
    }(log, mrow));

    // Run the pipeline on a dedicated thread.
    std::thread thd(run_actor_network, std::move(actors));

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
    while (log.size() < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
        std::lock_guard lk{log.mutex};
        // Smaller `net_memory_delta` has higher priority, so request 1 should complete
        // first.
        EXPECT_EQ(log.log.at(0).first, 1);
    }

    // Now allow the second request to complete.
    set_mem_avail(20);
    thd.join();
    {
        std::lock_guard lk{log.mutex};
        EXPECT_EQ(log.log.at(1).first, 2);
    }
}

TEST_P(StreamingMemoryReserveOrWait, RestartPeriodicTask) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    }

    MemoryReserveOrWait mrow{
        // Use a very high timeout to effectively disable timeout in this test.
        config::Options({{"memory_reserve_timeout", config::OptionValue("1 min")}}),
        MemoryType::DEVICE,
        ctx->executor(),
        ctx->br()
    };

    // Round 1: create a request, then make memory available.
    set_mem_avail(0);
    std::vector<Actor> actors1;
    actors1.push_back([](MemoryReserveOrWait& mrow) -> Actor {
        auto res = co_await mrow.reserve_or_wait(10, 0);
        EXPECT_EQ(res.size(), 10);
    }(mrow));

    std::thread thd1(run_actor_network, std::move(actors1));
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
    std::vector<Actor> actors2;
    actors2.push_back([](MemoryReserveOrWait& mrow) -> Actor {
        auto res = co_await mrow.reserve_or_wait(10, 0);
        EXPECT_EQ(res.size(), 10);
    }(mrow));

    std::thread thd2(run_actor_network, std::move(actors2));
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
        config::Options({{"memory_reserve_timeout", config::OptionValue("1 min")}}),
        MemoryType::DEVICE,
        ctx->executor(),
        ctx->br()
    };

    // Do multiple rounds to increase the chance we hit the "task exiting" window.
    for (int i = 0; i < 50; ++i) {
        set_mem_avail(0);
        std::vector<Actor> actors;
        actors.push_back([](MemoryReserveOrWait& mrow) -> Actor {
            auto res = co_await mrow.reserve_or_wait(10, 0);
            EXPECT_EQ(res.size(), 10);
        }(mrow));

        std::thread thd(run_actor_network, std::move(actors));

        while (mrow.size() < 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        set_mem_avail(20);
        thd.join();
    }
}

// Pins the memory available for reservation to `available` bytes and registers a spill
// function backed by a finite pool of `spillable` bytes, recording each request. Frees
// what it is asked for until the pool is exhausted, then returns 0.
//
// Pinning keeps the buffer resource's periodic spill thread out of the way, it only
// acts once the available memory goes negative.
class SpillRecorder {
  public:
    SpillRecorder(BufferResource* br, std::int64_t available, std::size_t spillable)
        : br_{br},
          limit_{
              safe_cast<std::int64_t>(br->device_mr_adaptor().current_allocated())
              + available
          },
          spillable_{spillable} {
        br_->set_memory_limit(MemoryType::DEVICE, limit_);
        fid_ = br_->spill_manager().add_spill_function(
            [this](std::size_t amount) -> std::size_t {
                std::lock_guard lock(mutex_);
                amounts_.push_back(amount);
                auto const spilled = std::min(amount, spillable_);
                spillable_ -= spilled;
                limit_ += safe_cast<std::int64_t>(spilled);
                br_->set_memory_limit(MemoryType::DEVICE, limit_);
                return spilled;
            },
            /* priority = */ 1
        );
    }

    ~SpillRecorder() {
        br_->spill_manager().remove_spill_function(fid_);
    }

    [[nodiscard]] std::vector<std::size_t> amounts() const {
        std::lock_guard lock(mutex_);
        return amounts_;
    }

  private:
    BufferResource* br_;
    std::int64_t limit_;
    std::size_t spillable_;
    std::size_t fid_{};
    mutable std::mutex mutex_;
    std::vector<std::size_t> amounts_;
};

TEST_P(StreamingMemoryReserveOrWait, AdmissionLoopNeverSpills) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    }

    // No periodic spill thread, so the admission loop is the only candidate spiller.
    auto br_no_periodic = make_br_without_periodic_spill(/* limit = */ 0);
    MemoryReserveOrWait mrow{
        // Keep the timeout far away so the test exercises ordinary admission polling.
        config::Options({{"memory_reserve_timeout", config::OptionValue("1 min")}}),
        MemoryType::DEVICE,
        ctx->executor(),
        br_no_periodic
    };

    // An outstanding reservation consumes all available memory. Spilling would unblock
    // the waiter, but the admission loop must not do it: that would stall the shared
    // executor, which is what #23892 and #1164 were about.
    SpillRecorder spills{
        br_no_periodic.get(), /* available = */ 10, /* spillable = */ 10
    };
    auto [outstanding, _] =
        br_no_periodic->reserve(MemoryType::DEVICE, 10, AllowOverbooking::NO);
    ASSERT_EQ(outstanding.size(), 10);

    std::vector<Actor> actors;
    actors.push_back([](MemoryReserveOrWait& waiter) -> Actor {
        try {
            std::ignore = co_await waiter.reserve_or_wait(10, 0);
        } catch (std::runtime_error const&) {
            // `shutdown()` closes the pending request queue.
        }
    }(mrow));
    actors.push_back(
        [](MemoryReserveOrWait& waiter, SpillRecorder const& spills) -> Actor {
            // The counter increments before each yield. Reaching two iterations proves
            // the first no-fit admission pass completed without relying on wall-clock
            // sleeps or a scheduling deadline.
            while (waiter.periodic_memory_check_counter() < 2) {
                co_await waiter.executor()->yield();
            }
            EXPECT_TRUE(spills.amounts().empty());
            co_await waiter.shutdown();
        }(mrow, spills)
    );
    run_actor_network(std::move(actors));
}

TEST_P(StreamingMemoryReserveOrWait, ProgressTimeoutReturnsWithoutSpilling) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    }

    // No periodic spill thread, so the timeout path is the only candidate spiller.
    auto br_no_periodic = make_br_without_periodic_spill(/* limit = */ 0);
    MemoryReserveOrWait mrow{
        // Short timeout, the waiter can only make progress via the timeout path.
        config::Options({{"memory_reserve_timeout", config::OptionValue("100ms")}}),
        MemoryType::DEVICE,
        ctx->executor(),
        br_no_periodic
    };

    // A timeout must preserve its bounded-progress contract by handing back a
    // zero-size reservation. It must not evict queued device data merely to turn
    // that timeout into an immediate full reservation.
    SpillRecorder spills{br_no_periodic.get(), /* available = */ 0, /* spillable = */ 10};
    ASSERT_EQ(br_no_periodic->memory_available(MemoryType::DEVICE), 0);

    // The waiter completes via the timeout instead of hanging on its queue.
    std::vector<Actor> actors;
    actors.push_back(waiter(mrow, 10, 0, 0));
    run_actor_network(std::move(actors));

    EXPECT_TRUE(spills.amounts().empty());
}

TEST_P(StreamingMemoryReserveOrWait, PeriodicThreadSpillsForWaitingRequest) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    }

    // The fixture's buffer resource runs the periodic spill thread, which is allowed to
    // spill on a waiter's behalf because it is not the executor. The admission loop
    // still never spills, see `AdmissionLoopNeverSpills`.
    MemoryReserveOrWait mrow{
        // So long that completing at all proves the timeout path was not what freed it.
        config::Options({{"memory_reserve_timeout", config::OptionValue("1 min")}}),
        MemoryType::DEVICE,
        ctx->executor(),
        ctx->br()
    };

    // No memory available, but enough is spillable to satisfy the request.
    SpillRecorder spills{br.get(), /* available = */ 0, /* spillable = */ 10};

    std::vector<Actor> actors;
    actors.push_back(waiter(mrow, 10, 0, 10));  // a full reservation, not a timeout
    run_actor_network(std::move(actors));

    EXPECT_FALSE(spills.amounts().empty())
        << "the periodic thread should have spilled for the waiting request";
}

TEST_P(StreamingMemoryReserveOrWait, NoSpillWhenMemoryIsAvailable) {
    MemoryReserveOrWait mrow{
        config::Options({{"memory_reserve_timeout", config::OptionValue("1 min")}}),
        MemoryType::DEVICE,
        ctx->executor(),
        ctx->br()
    };

    SpillRecorder spills{br.get(), /* available = */ 1024, /* spillable = */ 0};

    std::vector<Actor> actors;
    actors.push_back(waiter(mrow, 10, 0, 10));
    run_actor_network(std::move(actors));

    // The request fits immediately, so the fast path never reaches the periodic task.
    EXPECT_TRUE(spills.amounts().empty());
}

TEST_P(StreamingMemoryReserveOrWait, OverbookOnTimeoutReportsOverbookingBytes) {
    // Start with no available memory so the request cannot be satisfied normally.
    set_mem_avail(0);

    coro::sync_wait([](std::shared_ptr<Context> ctx) -> Actor {
        MemoryReserveOrWait mrow{
            // Use a very small timeout to trigger timeout immediately.
            config::Options({{"memory_reserve_timeout", config::OptionValue("1ns")}}),
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

    coro::sync_wait([](std::shared_ptr<Context> ctx) -> Actor {
        MemoryReserveOrWait mrow{
            // Use a very small timeout to trigger timeout immediately.
            config::Options({{"memory_reserve_timeout", config::OptionValue("1ns")}}),
            MemoryType::DEVICE,
            ctx->executor(),
            ctx->br()
        };
        EXPECT_THROW(
            std::ignore = co_await mrow.reserve_or_wait_or_fail(10, 0),
            rapidsmpf::reservation_error
        );
    }(ctx));
}

TEST_P(StreamingMemoryReserveOrWait, ReserveMemoryHelperWithOverbookingEnabled) {
    // Start with no available memory so the request cannot be satisfied normally.
    set_mem_avail(0);

    coro::sync_wait([](std::shared_ptr<Context> ctx) -> Actor {
        // Request should succeed with overbooking enabled.
        auto res = co_await reserve_memory(
            ctx,
            512,
            0,  // net_memory_delta
            MemoryType::DEVICE,
            AllowOverbooking::YES
        );
        EXPECT_EQ(res.mem_type(), MemoryType::DEVICE);
        EXPECT_EQ(res.size(), 512);
    }(ctx));
}

TEST_P(StreamingMemoryReserveOrWait, ReserveMemoryHelperWithOverbookingDisabled) {
    // Start with no available memory so the request cannot be satisfied.
    set_mem_avail(0);

    coro::sync_wait([](std::shared_ptr<Context> ctx) -> Actor {
        // Request should fail with overbooking disabled.
        EXPECT_THROW(
            std::ignore = co_await reserve_memory(
                ctx,
                512,
                0,  // net_memory_delta
                MemoryType::DEVICE,
                AllowOverbooking::NO
            ),
            rapidsmpf::reservation_error
        );
    }(ctx));
}

TEST_P(StreamingMemoryReserveOrWait, ReserveMemoryHelperWhenMemoryAvailable) {
    // Make memory available.
    set_mem_avail(1024);

    coro::sync_wait([](std::shared_ptr<Context> ctx) -> Actor {
        // Request should succeed without overbooking.
        auto res = co_await reserve_memory(
            ctx,
            512,
            0,  // net_memory_delta
            MemoryType::DEVICE,
            AllowOverbooking::NO
        );
        EXPECT_EQ(res.mem_type(), MemoryType::DEVICE);
        EXPECT_EQ(res.size(), 512);
    }(ctx));
}

TEST_P(StreamingMemoryReserveOrWait, ReserveMemoryHelperDefaultOverbookingEnabled) {
    // Start with no available memory.
    set_mem_avail(0);

    // Create a new context with allow_overbooking_by_default set to true.
    config::Options options{
        {{"memory_reserve_timeout", config::OptionValue("1ns")},
         {"allow_overbooking_by_default", config::OptionValue("true")}}
    };
    auto ctx_with_overbook = std::make_shared<Context>(
        options, GlobalEnvironment->comm_->logger(), ctx->executor(), ctx->br()
    );

    coro::sync_wait([](std::shared_ptr<Context> ctx) -> Actor {
        // Request should succeed because default is to allow overbooking.
        auto res = co_await reserve_memory(
            ctx,
            2048,
            0,  // net_memory_delta
            MemoryType::DEVICE,
            std::nullopt  // Use default from configuration
        );
        EXPECT_EQ(res.mem_type(), MemoryType::DEVICE);
        EXPECT_EQ(res.size(), 2048);
    }(ctx_with_overbook));
}

TEST_P(StreamingMemoryReserveOrWait, ReserveMemoryHelperDefaultOverbookingDisabled) {
    // Start with no available memory.
    set_mem_avail(0);

    // Create a new context with allow_overbooking_by_default set to false.
    config::Options options{
        {{"memory_reserve_timeout", config::OptionValue("1ns")},
         {"allow_overbooking_by_default", config::OptionValue("false")}}
    };
    auto ctx_with_no_overbook = std::make_shared<Context>(
        options, GlobalEnvironment->comm_->logger(), ctx->executor(), ctx->br()
    );

    coro::sync_wait([](std::shared_ptr<Context> ctx) -> Actor {
        // Request should fail because default is to disallow overbooking.
        EXPECT_THROW(
            std::ignore = co_await reserve_memory(
                ctx,
                2048,
                0,  // net_memory_delta
                MemoryType::DEVICE,
                std::nullopt  // Use default from configuration
            ),
            rapidsmpf::reservation_error
        );
    }(ctx_with_no_overbook));
}

TEST_P(StreamingMemoryReserveOrWait, StatisticsRecordWaitAvoided) {
    auto [br_stats, stats] = make_br_with_stats(/* device_limit = */ 1024);
    MemoryReserveOrWait mrow{
        config::Options({{"memory_reserve_timeout", config::OptionValue("1 min")}}),
        MemoryType::DEVICE,
        ctx->executor(),
        br_stats
    };

    std::vector<Actor> actors;
    actors.push_back(waiter(mrow, 10, 0, 10));
    run_actor_network(std::move(actors));

    // The fast path satisfied the request, so it never reached the periodic task.
    auto const avoided = stats->get_stat("reserve-device-wait-avoided");
    EXPECT_EQ(avoided.count(), 1u);  // one lookup
    EXPECT_EQ(avoided.value(), 1.0);  // one hit
    EXPECT_EQ(stats->get_stat("reserve-device-request-bytes").value(), 10.0);
    // Nothing queued, so the queued-request hit rate was never sampled.
    EXPECT_THROW(
        std::ignore = stats->get_stat("reserve-device-wait-timeout"), std::out_of_range
    );
}

TEST_P(StreamingMemoryReserveOrWait, StatisticsRecordWaitSatisfied) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    }

    // No memory initially, so the request must queue rather than take the fast path.
    auto [br_stats, stats] = make_br_with_stats(/* device_limit = */ 0);
    MemoryReserveOrWait mrow{
        // Keep the timeout far away so only ordinary admission can complete this.
        config::Options({{"memory_reserve_timeout", config::OptionValue("1 min")}}),
        MemoryType::DEVICE,
        ctx->executor(),
        br_stats
    };

    std::vector<Actor> actors;
    actors.push_back(waiter(mrow, 10, 0, 10));
    // Release memory once the request is queued. Condition driven, no wall-clock sleep.
    actors.push_back([](MemoryReserveOrWait& mrow, BufferResource& br) -> Actor {
        while (mrow.size() < 1) {
            co_await mrow.executor()->yield();
        }
        br.set_memory_limit(MemoryType::DEVICE, 10);
    }(mrow, *br_stats));
    run_actor_network(std::move(actors));

    // Queued, then admitted by the release rather than by the timeout.
    auto const avoided = stats->get_stat("reserve-device-wait-avoided");
    EXPECT_EQ(avoided.count(), 1u);  // one lookup
    EXPECT_EQ(avoided.value(), 0.0);  // no hit
    auto const timeout = stats->get_stat("reserve-device-wait-timeout");
    EXPECT_EQ(timeout.count(), 1u);  // one queued request
    EXPECT_EQ(timeout.value(), 0.0);  // admitted by the release, it never timed out
    // One request pending, counting itself.
    EXPECT_EQ(stats->get_stat("reserve-device-waiting-requests").max(), 1.0);
    EXPECT_EQ(stats->get_stat("reserve-device-wait-satisfied-time").count(), 1u);
    EXPECT_THROW(
        std::ignore = stats->get_stat("reserve-device-wait-timeout-time"),
        std::out_of_range
    );
}

TEST_P(StreamingMemoryReserveOrWait, StatisticsRecordWaitTimeout) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    }

    // No memory, and none is ever released, so only the timeout path can complete it.
    auto [br_stats, stats] = make_br_with_stats(/* device_limit = */ 0);
    MemoryReserveOrWait mrow{
        config::Options({{"memory_reserve_timeout", config::OptionValue("100ms")}}),
        MemoryType::DEVICE,
        ctx->executor(),
        br_stats
    };

    std::vector<Actor> actors;
    actors.push_back(waiter(mrow, 10, 0, 0));  // zero-size reservation
    run_actor_network(std::move(actors));

    auto const timeout = stats->get_stat("reserve-device-wait-timeout");
    EXPECT_EQ(timeout.count(), 1u);  // one queued request
    EXPECT_EQ(timeout.value(), 1.0);  // it ran out the timeout
    // The loop only breaks once more than `timeout_` has elapsed, so the recorded
    // wait cannot be shorter than the timeout. Compared against half of it to leave
    // room for clock granularity.
    EXPECT_GE(stats->get_stat("reserve-device-wait-timeout-time").value(), 0.05);
    EXPECT_THROW(
        std::ignore = stats->get_stat("reserve-device-wait-satisfied-time"),
        std::out_of_range
    );
}

TEST_P(StreamingMemoryReserveOrWait, StatisticsRecordOverbooking) {
    auto [br_stats, stats] = make_br_with_stats(/* device_limit = */ 0);
    MemoryReserveOrWait mrow{
        // A tiny timeout so the request reaches the overbooking fallback at once.
        config::Options({{"memory_reserve_timeout", config::OptionValue("1ns")}}),
        MemoryType::DEVICE,
        ctx->executor(),
        br_stats
    };

    coro::sync_wait([](MemoryReserveOrWait& mrow) -> Actor {
        // Both reservations are held for the duration, so the second one overbooks on
        // top of the first rather than starting from a clean slate.
        auto [first, first_overbooked] = co_await mrow.reserve_or_wait_or_overbook(10, 0);
        EXPECT_EQ(first.size(), 10);
        EXPECT_EQ(first_overbooked, 10);

        auto [second, second_overbooked] =
            co_await mrow.reserve_or_wait_or_overbook(10, 0);
        EXPECT_EQ(second.size(), 10);
        // `reserve()` reports the total deficit, which now includes the first
        // reservation as well.
        EXPECT_EQ(second_overbooked, 20);
    }(mrow));

    auto const overbooked = stats->get_stat("reserve-device-overbook-bytes");
    EXPECT_EQ(overbooked.count(), 2u);
    // 10 each. Recording the raw `reserve()` result instead would double count the
    // first reservation and give 30.
    EXPECT_EQ(overbooked.value(), 20.0);
    EXPECT_EQ(overbooked.max(), 10.0);
}

TEST_P(StreamingMemoryReserveOrWait, StatisticsDisabledRecordsNothing) {
    // The fixture's buffer resource carries `Statistics::disabled()`.
    auto stats = ctx->br()->statistics();
    ASSERT_FALSE(stats->enabled());

    MemoryReserveOrWait mrow{
        config::Options({{"memory_reserve_timeout", config::OptionValue("1 min")}}),
        MemoryType::DEVICE,
        ctx->executor(),
        ctx->br()
    };
    set_mem_avail(1024);

    std::vector<Actor> actors;
    actors.push_back(waiter(mrow, 10, 0, 10));  // behaviour is unchanged
    run_actor_network(std::move(actors));

    EXPECT_TRUE(stats->list_stat_names().empty());
}
