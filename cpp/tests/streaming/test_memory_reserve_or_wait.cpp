/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <condition_variable>
#include <limits>
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
    MemoryType mem_type;
};

namespace {

/// @brief Whether the buffer resource can be set up for `mem_type`.
bool is_supported(MemoryType mem_type) {
    return mem_type != MemoryType::PINNED_HOST || is_pinned_memory_resources_supported();
}

std::optional<PinnedPoolProperties> pinned_properties(MemoryType mem_type) {
    return mem_type == MemoryType::PINNED_HOST
               ? std::optional<PinnedPoolProperties>{PinnedPoolProperties{}}
               : PinnedMemoryDisabled;
}

/// @brief Bytes currently allocated through the buffer resource for `mem_type`.
std::int64_t allocated(BufferResource& br, MemoryType mem_type) {
    switch (mem_type) {
    case MemoryType::DEVICE:
        return br.device_mr_adaptor().current_allocated();
    case MemoryType::PINNED_HOST:
        return br.try_pinned_mr()->current_allocated();
    case MemoryType::HOST:
        return br.host_mr_adaptor().current_allocated();
    case MemoryType::DISK:
        break;
    }
    RAPIDSMPF_FAIL("allocation is not tracked for " + std::string{to_string(mem_type)});
}

}  // namespace

class StreamingMemoryReserveOrWait
    : public BaseStreamingFixture,
      public ::testing::WithParamInterface<ReserveOrWaitParam> {
  public:
    void SetUp() override {
        // Keep the multi-rank runs short, these tests don't depend on the communicator.
        if (GlobalEnvironment->type() != TestEnvironmentType::SINGLE) {
            GTEST_SKIP() << "Runs only in the single-process environment";
        }
        if (!is_supported(mem_type())) {
            GTEST_SKIP() << to_string(mem_type()) << " memory is not supported";
        }
        // Drive memory availability by mutating the limit of the memory type under
        // test. No real allocations occur in these tests, so `memory_available()`
        // equals whatever limit we set.
        SetUpWithThreads(
            GetParam().num_threads, pinned_properties(mem_type()), {{mem_type(), 0}}
        );
    }

  protected:
    /// @brief The memory type under test.
    static MemoryType mem_type() {
        return GetParam().mem_type;
    }

    /// @brief Name of a statistic recorded for the memory type under test.
    static std::string stat_name(std::string_view suffix) {
        return "reserve-" + to_lower(to_string(mem_type())) + "-" + std::string{suffix};
    }

    void set_mem_avail(std::int64_t size) {
        br->set_memory_limit(mem_type(), size);
    }

    std::int64_t get_mem_avail() {
        return br->memory_available(mem_type());
    }

    // Buffer resource with statistics enabled and no periodic spill thread, so the
    // recorded stats come only from the reservation path under test. The fixture's
    // own `br` uses `Statistics::disabled()`.
    struct StatsBufferResource {
        std::shared_ptr<BufferResource> br;
        std::shared_ptr<Statistics> stats;
    };

    StatsBufferResource make_br_with_stats(std::int64_t limit) {
        auto stats = Statistics::create();
        auto br_with_stats = BufferResource::create(
            mr_cuda,
            pinned_properties(mem_type()),
            {{mem_type(), limit}},
            /* periodic_spill_check = */ std::nullopt,
            std::make_shared<StreamPool>(16),
            stats
        );
        return {.br = std::move(br_with_stats), .stats = std::move(stats)};
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
    ::testing::ValuesIn([] {
        std::vector<ReserveOrWaitParam> params;
        for (int num_threads : {1, 2, 8}) {
            for (auto mem_type : ADDRESSABLE_MEMORY_TYPES) {
                params.push_back({num_threads, mem_type});
            }
        }
        return params;
    }()),
    [](::testing::TestParamInfo<ReserveOrWaitParam> const& info) {
        return std::to_string(info.param.num_threads) + "threads_"
               + to_string(info.param.mem_type);
    }
);

TEST_P(StreamingMemoryReserveOrWait, AccessorsReturnExpectedValues) {
    config::Options options{
        {{"memory_reserve_timeout", config::OptionValue("12345 ms")}}
    };

    MemoryReserveOrWait mrow{options, mem_type(), ctx->executor(), ctx->br()};

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
        mem_type(),
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
        mem_type(),
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
        mem_type(),
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
        mem_type(),
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
    SpillRecorder(
        BufferResource* br,
        std::int64_t available,
        std::size_t spillable,
        MemoryType mem_type
    )
        : br_{br}, mem_type_{mem_type}, spillable_{spillable} {
        limit_ = allocated(*br_, mem_type_) + available;
        br_->set_memory_limit(mem_type_, limit_);
        fid_ = br_->spill_manager().add_spill_function(
            [this](std::size_t amount, MemoryType) -> std::size_t {
                std::lock_guard lock(mutex_);
                amounts_.push_back(amount);
                auto const spilled = std::min(amount, spillable_);
                spillable_ -= spilled;
                limit_ += safe_cast<std::int64_t>(spilled);
                br_->set_memory_limit(mem_type_, limit_);
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
    MemoryType mem_type_;
    std::int64_t limit_{0};
    std::size_t spillable_;
    std::size_t fid_{};
    mutable std::mutex mutex_;
    std::vector<std::size_t> amounts_;
};

TEST_P(StreamingMemoryReserveOrWait, DoesNotSpillBeforeProgressTimeout) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    }

    MemoryReserveOrWait mrow{
        // Keep the timeout far away so the test exercises ordinary admission polling.
        config::Options({{"memory_reserve_timeout", config::OptionValue("1 min")}}),
        mem_type(),
        ctx->executor(),
        ctx->br()
    };

    // An outstanding reservation consumes all available memory. Before the fix, the
    // ordinary admission loop calls this spill function to unblock the waiter.
    SpillRecorder spills{
        br.get(), /* available = */ 10, /* spillable = */ 10, mem_type()
    };
    auto [outstanding, _] = br->reserve(mem_type(), 10, AllowOverbooking::NO);
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

    MemoryReserveOrWait mrow{
        // Short timeout, the waiter can only make progress via the timeout path.
        config::Options({{"memory_reserve_timeout", config::OptionValue("100ms")}}),
        mem_type(),
        ctx->executor(),
        ctx->br()
    };

    // A timeout must preserve its bounded-progress contract by handing back a
    // zero-size reservation. It must not evict queued device data merely to turn
    // that timeout into an immediate full reservation.
    SpillRecorder spills{br.get(), /* available = */ 0, /* spillable = */ 10, mem_type()};
    ASSERT_EQ(get_mem_avail(), 0);

    // The waiter completes via the timeout instead of hanging on its queue.
    std::vector<Actor> actors;
    actors.push_back(waiter(mrow, 10, 0, 0));
    run_actor_network(std::move(actors));

    EXPECT_TRUE(spills.amounts().empty());
}

TEST_P(StreamingMemoryReserveOrWait, NoSpillWhenMemoryIsAvailable) {
    MemoryReserveOrWait mrow{
        config::Options({{"memory_reserve_timeout", config::OptionValue("1 min")}}),
        mem_type(),
        ctx->executor(),
        ctx->br()
    };

    SpillRecorder spills{
        br.get(), /* available = */ 1024, /* spillable = */ 0, mem_type()
    };

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
            mem_type(),
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
            mem_type(),
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
            mem_type(),
            AllowOverbooking::YES
        );
        EXPECT_EQ(res.mem_type(), mem_type());
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
                mem_type(),
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
            mem_type(),
            AllowOverbooking::NO
        );
        EXPECT_EQ(res.mem_type(), mem_type());
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
            mem_type(),
            std::nullopt  // Use default from configuration
        );
        EXPECT_EQ(res.mem_type(), mem_type());
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
                mem_type(),
                std::nullopt  // Use default from configuration
            ),
            rapidsmpf::reservation_error
        );
    }(ctx_with_no_overbook));
}

TEST_P(StreamingMemoryReserveOrWait, StatisticsRecordWaitAvoided) {
    auto [br_stats, stats] = make_br_with_stats(/* limit = */ 1024);
    MemoryReserveOrWait mrow{
        config::Options({{"memory_reserve_timeout", config::OptionValue("1 min")}}),
        mem_type(),
        ctx->executor(),
        br_stats
    };

    std::vector<Actor> actors;
    actors.push_back(waiter(mrow, 10, 0, 10));
    run_actor_network(std::move(actors));

    // The fast path satisfied the request, so it never reached the periodic task.
    auto const avoided = stats->get_stat(stat_name("wait-avoided"));
    EXPECT_EQ(avoided.count(), 1u);  // one lookup
    EXPECT_EQ(avoided.value(), 1.0);  // one hit
    EXPECT_EQ(stats->get_stat(stat_name("request-bytes")).value(), 10.0);
    // Nothing queued, so the queued-request hit rate was never sampled.
    EXPECT_THROW(
        std::ignore = stats->get_stat(stat_name("wait-timeout")), std::out_of_range
    );
}

TEST_P(StreamingMemoryReserveOrWait, StatisticsRecordWaitSatisfied) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    }

    // No memory initially, so the request must queue rather than take the fast path.
    auto [br_stats, stats] = make_br_with_stats(/* limit = */ 0);
    MemoryReserveOrWait mrow{
        // Keep the timeout far away so only ordinary admission can complete this.
        config::Options({{"memory_reserve_timeout", config::OptionValue("1 min")}}),
        mem_type(),
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
        br.set_memory_limit(mem_type(), 10);
    }(mrow, *br_stats));
    run_actor_network(std::move(actors));

    // Queued, then admitted by the release rather than by the timeout.
    auto const avoided = stats->get_stat(stat_name("wait-avoided"));
    EXPECT_EQ(avoided.count(), 1u);  // one lookup
    EXPECT_EQ(avoided.value(), 0.0);  // no hit
    auto const timeout = stats->get_stat(stat_name("wait-timeout"));
    EXPECT_EQ(timeout.count(), 1u);  // one queued request
    EXPECT_EQ(timeout.value(), 0.0);  // admitted by the release, it never timed out
    // One request pending, counting itself.
    EXPECT_EQ(stats->get_stat(stat_name("waiting-requests")).max(), 1.0);
    EXPECT_EQ(stats->get_stat(stat_name("wait-satisfied-time")).count(), 1u);
    EXPECT_THROW(
        std::ignore = stats->get_stat(stat_name("wait-timeout-time")), std::out_of_range
    );
}

TEST_P(StreamingMemoryReserveOrWait, StatisticsRecordWaitTimeout) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    }

    // No memory, and none is ever released, so only the timeout path can complete it.
    auto [br_stats, stats] = make_br_with_stats(/* limit = */ 0);
    MemoryReserveOrWait mrow{
        config::Options({{"memory_reserve_timeout", config::OptionValue("100ms")}}),
        mem_type(),
        ctx->executor(),
        br_stats
    };

    std::vector<Actor> actors;
    actors.push_back(waiter(mrow, 10, 0, 0));  // zero-size reservation
    run_actor_network(std::move(actors));

    auto const timeout = stats->get_stat(stat_name("wait-timeout"));
    EXPECT_EQ(timeout.count(), 1u);  // one queued request
    EXPECT_EQ(timeout.value(), 1.0);  // it ran out the timeout
    // The loop only breaks once more than `timeout_` has elapsed, so the recorded
    // wait cannot be shorter than the timeout. Compared against half of it to leave
    // room for clock granularity.
    EXPECT_GE(stats->get_stat(stat_name("wait-timeout-time")).value(), 0.05);
    EXPECT_THROW(
        std::ignore = stats->get_stat(stat_name("wait-satisfied-time")), std::out_of_range
    );
}

TEST_P(StreamingMemoryReserveOrWait, StatisticsRecordOverbooking) {
    auto [br_stats, stats] = make_br_with_stats(/* limit = */ 0);
    MemoryReserveOrWait mrow{
        // A tiny timeout so the request reaches the overbooking fallback at once.
        config::Options({{"memory_reserve_timeout", config::OptionValue("1ns")}}),
        mem_type(),
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
        // `reserve()` reports the total overbooking, which now includes the first
        // reservation as well.
        EXPECT_EQ(second_overbooked, 20);
    }(mrow));

    auto const overbooked = stats->get_stat(stat_name("overbook-bytes"));
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
        mem_type(),
        ctx->executor(),
        ctx->br()
    };
    set_mem_avail(1024);

    std::vector<Actor> actors;
    actors.push_back(waiter(mrow, 10, 0, 10));  // behaviour is unchanged
    run_actor_network(std::move(actors));

    EXPECT_TRUE(stats->list_stat_names().empty());
}

namespace {

/// @brief Spill function whose execution the test controls.
///
/// Each spill blocks inside the spill function until released, so a test can hold a
/// spill "in flight" for as long as it needs, then frees `frees` bytes by raising the
/// limit of `mem_type`. Only what `SpillManager` observes matters here, no device memory
/// moves. Spills run one at a time, so start the next only once the previous one has
/// finished.
class BlockingSpill {
  public:
    BlockingSpill(BufferResource* br, std::size_t frees, MemoryType mem_type)
        : br_{br}, mem_type_{mem_type}, frees_{frees} {
        limit_ = allocated(*br_, mem_type_);
        br_->set_memory_limit(mem_type_, limit_);
        fid_ = br_->spill_manager().add_spill_function(
            [this](std::size_t, MemoryType) -> std::size_t {
                {
                    std::unique_lock lock(mutex_);
                    int const me = ++entered_;
                    cv_.notify_all();
                    cv_.wait(lock, [&] { return released_ >= me; });
                }
                limit_ += safe_cast<std::int64_t>(frees_);
                br_->set_memory_limit(mem_type_, limit_);
                return frees_;
            },
            /* priority = */ 1
        );
    }

    ~BlockingSpill() {
        {
            std::lock_guard lock(mutex_);
            released_ = std::numeric_limits<int>::max();
        }
        cv_.notify_all();
        for (auto& spiller : spillers_) {
            spiller.join();
        }
        br_->spill_manager().remove_spill_function(fid_);
    }

    /// @brief Start another spill on its own thread and return once it is executing.
    void start() {
        spillers_.emplace_back([this] {
            std::ignore = br_->spill_manager().spill(1, mem_type_);
        });
        auto const n = static_cast<int>(spillers_.size());
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [&] { return entered_ >= n; });
    }

    /// @brief Let the oldest spill that is still held finish.
    void release_one() {
        {
            std::lock_guard lock(mutex_);
            ++released_;
        }
        cv_.notify_all();
    }

    /// @brief Let the oldest held spill finish, then start the next one.
    ///
    /// Blocks rather than yields, so called from an actor on a single-threaded executor
    /// it keeps the admission loop from running until the next spill is in flight.
    void finish_one_and_start_next() {
        auto& spill_manager = br_->spill_manager();
        auto const generation = spill_manager.spill_generation(mem_type_);
        release_one();
        while (spill_manager.spill_generation(mem_type_) == generation) {
            std::this_thread::yield();
        }
        start();
    }

  private:
    BufferResource* br_;
    MemoryType mem_type_;
    std::int64_t limit_{0};
    std::size_t frees_;
    std::size_t fid_{};
    std::vector<std::thread> spillers_;
    std::mutex mutex_;
    std::condition_variable cv_;
    int entered_{0};
    int released_{0};
};

/// @brief Yield until the admission loop has completed a full pass from now.
///
/// The loop counts an iteration before yielding and checks the deadline after, so the
/// count advancing by two means one whole pass has run. Returns early once nothing is
/// pending, since the loop then stops counting, so a request answered early fails the
/// test rather than hanging it.
Actor until_checked_again(MemoryReserveOrWait& w) {
    auto const count = w.periodic_memory_check_counter();
    while (w.size() > 0 && w.periodic_memory_check_counter() < count + 2) {
        co_await w.executor()->yield();
    }
}

/// @brief Yield until the admission loop has checked a queued request past its deadline.
///
/// Elapsed time is only ever a lower bound, so being descheduled can delay this but
/// never cut it short.
Actor until_checked_past_deadline(MemoryReserveOrWait& w) {
    // Once the loop has run, the request is queued and its deadline is set.
    while (w.periodic_memory_check_counter() < 1) {
        co_await w.executor()->yield();
    }
    auto const queued = Clock::now();
    while (Clock::now() - queued <= w.timeout()) {
        co_await w.executor()->yield();
    }
    co_await until_checked_again(w);
}

}  // namespace

TEST_P(StreamingMemoryReserveOrWait, SpillInFlightExtendsThePastDueDeadline) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    }

    auto [br_stats, stats] = make_br_with_stats(/* limit = */ 0);
    MemoryReserveOrWait mrow{
        config::Options({{"memory_reserve_timeout", config::OptionValue("200ms")}}),
        mem_type(),
        ctx->executor(),
        br_stats
    };

    // A spill is running and will free exactly what the waiter needs.
    BlockingSpill spill{br_stats.get(), /* frees = */ 10, mem_type()};
    spill.start();

    std::vector<Actor> actors;
    // Admitted with a full reservation rather than the zero-size timeout result.
    actors.push_back(waiter(mrow, 10, 0, 10));
    actors.push_back([](MemoryReserveOrWait& w, BlockingSpill& s) -> Actor {
        // Release only once the loop has seen the held spill past the deadline, so it
        // has extended and the request can only be admitted through that extension.
        co_await until_checked_past_deadline(w);
        s.release_one();
    }(mrow, spill));
    run_actor_network(std::move(actors));

    auto const extended = stats->get_stat(stat_name("wait-spill-extended"));
    EXPECT_EQ(extended.count(), 1u);
    EXPECT_EQ(extended.value(), 1.0);
    auto const rescued = stats->get_stat(stat_name("wait-spill-rescued"));
    EXPECT_EQ(rescued.count(), 1u);
    EXPECT_EQ(rescued.value(), 1.0);
    EXPECT_GT(stats->get_stat(stat_name("wait-spill-extension-time")).value(), 0.0);
    // Admitted, so it never took the timeout exit.
    EXPECT_EQ(stats->get_stat(stat_name("wait-timeout")).value(), 0.0);
}

TEST_P(StreamingMemoryReserveOrWait, NoExtensionWhenNothingIsSpilling) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    }

    auto [br_stats, stats] = make_br_with_stats(/* limit = */ 0);
    MemoryReserveOrWait mrow{
        config::Options({{"memory_reserve_timeout", config::OptionValue("10ms")}}),
        mem_type(),
        ctx->executor(),
        br_stats
    };

    std::vector<Actor> actors;
    actors.push_back(waiter(mrow, 10, 0, 0));
    run_actor_network(std::move(actors));

    // With no spill running there is no evidence that memory is coming, so the timeout
    // stands.
    auto const extended = stats->get_stat(stat_name("wait-spill-extended"));
    EXPECT_EQ(extended.count(), 1u);
    EXPECT_EQ(extended.value(), 0.0);
    EXPECT_EQ(stats->get_stat(stat_name("wait-timeout")).value(), 1.0);
    EXPECT_THROW(
        std::ignore = stats->get_stat(stat_name("wait-spill-rescued")), std::out_of_range
    );
}

TEST_P(StreamingMemoryReserveOrWait, AnUnproductiveSpillEndsTheExtensionForGood) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    }
    if (GetParam().num_threads != 1) {
        // The test holds the only executor thread to freeze the admission loop while it
        // swaps one running spill for the next, which needs exactly one thread.
        GTEST_SKIP() << "Needs a single executor thread";
    }

    auto [br_stats, stats] = make_br_with_stats(/* limit = */ 0);
    MemoryReserveOrWait mrow{
        config::Options({{"memory_reserve_timeout", config::OptionValue("200ms")}}),
        mem_type(),
        ctx->executor(),
        br_stats
    };

    // Spills that free nothing, each held open until the test releases it.
    BlockingSpill spill{br_stats.get(), /* frees = */ 0, mem_type()};
    spill.start();

    std::vector<Actor> actors;
    actors.push_back(waiter(mrow, 10, 0, 0));
    actors.push_back([](MemoryReserveOrWait& w, BlockingSpill& s) -> Actor {
        // Let the loop see the first spill past the deadline, so it extends on it.
        co_await until_checked_past_deadline(w);
        // Now hold the only executor thread, so the loop cannot run, while the first
        // spill completes having freed nothing and a second one starts. When the loop
        // next looks, it sees both a completed unproductive spill and one in flight.
        s.finish_one_and_start_next();
        // The completed spill freed nothing, so the verdict is final even though another
        // spill is running. One pass judges the unproductive spill, the next gives up.
        // The second spill is still held, so a re-extension would leave the request
        // queued. Two calls are needed because `until_checked_again()` can return right
        // after the loop counts the second pass, before that pass gives up.
        co_await until_checked_again(w);
        co_await until_checked_again(w);
        EXPECT_EQ(w.size(), 0u);
    }(mrow, spill));
    run_actor_network(std::move(actors));

    EXPECT_EQ(stats->get_stat(stat_name("wait-spill-rescued")).value(), 0.0);
}

/// @brief Cross-type checks. Not parameterized, since they pick their own memory types.
class StreamingMemoryReserveOrWaitCrossType : public BaseStreamingFixture {
  public:
    void SetUp() override {
        if (is_running_under_valgrind()) {
            GTEST_SKIP() << "Test runs very slow in valgrind";
        }
        if (GlobalEnvironment->type() != TestEnvironmentType::SINGLE) {
            GTEST_SKIP() << "Runs only in the single-process environment";
        }
        BaseStreamingFixture::SetUp();
    }
};

TEST_F(StreamingMemoryReserveOrWaitCrossType, SpillOfAnotherMemoryTypeDoesNotExtend) {
    auto stats = Statistics::create();
    auto br_stats = BufferResource::create(
        mr_cuda,
        rapidsmpf::PinnedMemoryDisabled,
        {{MemoryType::DEVICE, 0}, {MemoryType::HOST, 0}},
        /* periodic_spill_check = */ std::nullopt,
        std::make_shared<StreamPool>(16),
        stats
    );
    MemoryReserveOrWait mrow{
        config::Options({{"memory_reserve_timeout", config::OptionValue("10ms")}}),
        MemoryType::HOST,
        ctx->executor(),
        br_stats
    };

    // A device spill is running and never finishes. It frees no host memory, it only
    // moves data into it, so a host request has nothing to wait for.
    BlockingSpill spill{br_stats.get(), /* frees = */ 0, MemoryType::DEVICE};
    spill.start();

    std::vector<Actor> actors;
    actors.push_back([](MemoryReserveOrWait& w) -> Actor {
        auto res = co_await w.reserve_or_wait(10, 0);
        EXPECT_EQ(res.size(), 0u);
    }(mrow));
    run_actor_network(std::move(actors));

    // Timed out as usual, rather than extending on a spill that cannot help.
    auto const extended = stats->get_stat("reserve-host-wait-spill-extended");
    EXPECT_EQ(extended.count(), 1u);
    EXPECT_EQ(extended.value(), 0.0);
    EXPECT_EQ(stats->get_stat("reserve-host-wait-timeout").value(), 1.0);
}

TEST_P(StreamingMemoryReserveOrWait, AdmitsDuringAnExtensionOnceMemoryIsReleased) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    }

    auto [br_stats, stats] = make_br_with_stats(/* limit = */ 0);
    MemoryReserveOrWait mrow{
        config::Options({{"memory_reserve_timeout", config::OptionValue("200ms")}}),
        mem_type(),
        ctx->executor(),
        br_stats
    };

    // A spill that is held for the whole test, so the request extends and keeps
    // extending on it.
    BlockingSpill spill{br_stats.get(), /* frees = */ 0, mem_type()};
    spill.start();

    std::vector<Actor> actors;
    actors.push_back(waiter(mrow, 10, 0, 10));
    actors.push_back([](MemoryReserveOrWait& w) -> Actor {
        co_await until_checked_past_deadline(w);
        // Memory comes back through a release rather than the spill, which is still
        // running. The request fits now and should not wait for the spill to end.
        auto& br = *w.br();
        br.set_memory_limit(mem_type(), allocated(br, mem_type()) + 10);
        co_await until_checked_again(w);
        EXPECT_EQ(w.size(), 0u);
    }(mrow));
    run_actor_network(std::move(actors));

    // Counted as rescued even though the memory came from a release, since the
    // request was admitted during the extension instead of overbooking.
    auto const rescued = stats->get_stat(stat_name("wait-spill-rescued"));
    EXPECT_EQ(rescued.count(), 1u);
    EXPECT_EQ(rescued.value(), 1.0);
}

TEST_P(StreamingMemoryReserveOrWait, SpillProgressCountsWhileStillOverbooked) {
    if (is_running_under_valgrind()) {
        GTEST_SKIP() << "Test runs very slow in valgrind";
    }
    if (GetParam().num_threads != 1) {
        // The test holds the only executor thread to freeze the admission loop while it
        // swaps one running spill for the next, which needs exactly one thread.
        GTEST_SKIP() << "Needs a single executor thread";
    }

    auto br_stats = make_br_with_stats(/* limit = */ 0).br;
    MemoryReserveOrWait mrow{
        config::Options({{"memory_reserve_timeout", config::OptionValue("200ms")}}),
        mem_type(),
        ctx->executor(),
        br_stats
    };

    // Each spill frees 60 bytes. Another caller has overbooked by 100, so the first
    // spill leaves availability at -40, progress that is still below zero, and only the
    // second brings it to +20.
    BlockingSpill spill{br_stats.get(), /* frees = */ 60, mem_type()};
    auto overbooked = br_stats->reserve(mem_type(), 100, AllowOverbooking::YES);
    spill.start();

    std::vector<Actor> actors;
    // Admitted once the second spill lands, rather than overbooking after the first.
    actors.push_back(waiter(mrow, 10, 0, 10));
    actors.push_back([](MemoryReserveOrWait& w, BlockingSpill& s) -> Actor {
        co_await until_checked_past_deadline(w);
        // Hold the only executor thread while the first spill completes and a second
        // one starts, so the loop next judges a productive spill with another running.
        s.finish_one_and_start_next();
        co_await until_checked_again(w);
        s.release_one();
    }(mrow, spill));
    run_actor_network(std::move(actors));
}
