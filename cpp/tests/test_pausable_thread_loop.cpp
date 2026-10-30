/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <random>
#include <ranges>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cuda/stream>

#include <rmm/mr/cuda_memory_resource.hpp>

#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/pausable_thread_loop.hpp>
#include <rapidsmpf/utils/misc.hpp>

using rapidsmpf::detail::PausableThreadLoop;

TEST(PausableThreadLoop, ResumeAndPause) {
    int counter{0};
    std::mutex mutex;
    std::condition_variable cv;
    bool updated = false;

    PausableThreadLoop loop([&]() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++counter;
            updated = true;
        }
        cv.notify_one();
    });

    // The loop starts paused.
    EXPECT_FALSE(loop.is_running());
    loop.resume();
    EXPECT_TRUE(loop.is_running());

    // Wait for the counter to be increased.
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&]() { return updated; });
    }
    EXPECT_GT(counter, 0);

    // If we pause, the counter should stay the same.
    loop.pause();
    int count_after_pause = counter;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    loop.stop();
    // We accept counter-1 since the loop function might have passed the wait check when
    // we called pause().
    EXPECT_THAT(
        count_after_pause, testing::AnyOf(testing::Eq(counter), testing::Eq(counter - 1))
    );
}

TEST(PausableThreadLoop, MultiplePauseAndResume) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distr(5, 10);

    PausableThreadLoop loop([&] {
        // thread sleeps for a [5, 10) milliseconds
        std::this_thread::sleep_for(std::chrono::milliseconds(distr(gen)));
    });

    std::array<std::future<void>, 8> futures{
        std::async(std::launch::async, [&] { loop.pause(); }),
        std::async(std::launch::async, [&] { loop.resume(); }),
        std::async(std::launch::async, [&] { loop.pause_nb(); }),
        std::async(std::launch::async, [&] { loop.resume(); }),
        std::async(std::launch::async, [&] { loop.pause(); }),
        std::async(std::launch::async, [&] { loop.resume(); }),
        std::async(std::launch::async, [&] { loop.pause_nb(); }),
        std::async(std::launch::async, [&] { loop.resume(); })
    };

    std::ranges::for_each(futures, [](auto& f) { f.get(); });

    // loop could be running/paused. But all calls should have completed.
    loop.stop();
    EXPECT_FALSE(loop.is_running());
}

TEST(PausableThreadLoop, CanDoCudaWorkOnFirstTick) {
    using namespace rapidsmpf;

    if (!is_pinned_memory_resources_supported()) {
        GTEST_SKIP() << "Pinned memory not supported on this system";
    }

    rmm::mr::cuda_memory_resource cuda_mr;
    auto br = BufferResource::create(cuda_mr, PinnedPoolProperties{});
    auto stream = cuda::stream_ref{cudaStreamLegacy};

    std::exception_ptr eptr;
    std::promise<void> first_tick_done;
    auto first_tick_future = first_tick_done.get_future();
    bool signaled = false;
    PausableThreadLoop loop([&]() {
        try {
            auto [reservation, _] =
                br->reserve(MemoryType::PINNED_HOST, 1024, AllowOverbooking::YES);
            auto buf = br->make_buffer(1024, stream, reservation);
        } catch (...) {
            eptr = std::current_exception();
        }
        if (!signaled) {
            signaled = true;
            first_tick_done.set_value();
        }
    });
    loop.resume();
    // Wait for the first iteration to finish so that the CUDA work actually ran.
    ASSERT_EQ(
        first_tick_future.wait_for(std::chrono::seconds{30}), std::future_status::ready
    );
    loop.stop();

    if (eptr) {
        std::rethrow_exception(eptr);
    }
}

namespace {

// Sleep duration that the tests below must never wait out.
constexpr auto long_sleep = std::chrono::seconds{10};
// Bound for the first iteration, which includes setting up the thread's CUDA context.
constexpr auto setup_timeout = std::chrono::seconds{30};
// Bound for anything that must interrupt the sleep. Well below `long_sleep`, but
// generous enough for loaded CI runners and sanitizer builds.
constexpr auto prompt = std::chrono::seconds{5};
// Time given to the loop thread to go from the end of the loop function into its sleep.
constexpr auto enter_sleep = std::chrono::milliseconds{50};

/// Counts loop iterations and lets the test wait for a given count.
class IterationCounter {
  public:
    /// Records one iteration and returns the new count.
    int increment() {
        int count;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            count = ++count_;
        }
        cv_.notify_all();
        return count;
    }

    [[nodiscard]] int get() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    }

    /// Waits until at least `n` iterations have run. Returns false on timeout.
    bool wait_for(int n, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [&] { return count_ >= n; });
    }

  private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    int count_{0};
};

}  // namespace

TEST(PausableThreadLoop, ResumeInterruptsSleep) {
    IterationCounter counter;
    PausableThreadLoop loop([&] { counter.increment(); }, long_sleep);
    loop.resume();

    // The first iteration runs immediately, then the loop sleeps.
    ASSERT_TRUE(counter.wait_for(1, setup_timeout));
    EXPECT_EQ(counter.get(), 1);
    std::this_thread::sleep_for(enter_sleep);

    loop.resume();
    EXPECT_TRUE(counter.wait_for(2, prompt));
    loop.stop();
}

TEST(PausableThreadLoop, ResumeDuringIterationSkipsNextSleep) {
    IterationCounter counter;
    std::promise<void> resumed;
    auto resumed_future = resumed.get_future();
    PausableThreadLoop loop(
        [&] {
            // Hold the first iteration open until the test has called resume().
            if (counter.increment() == 1) {
                resumed_future.wait_for(long_sleep);
            }
        },
        long_sleep
    );
    loop.resume();
    ASSERT_TRUE(counter.wait_for(1, setup_timeout));

    // The first iteration is still running.
    loop.resume();
    resumed.set_value();
    EXPECT_TRUE(counter.wait_for(2, prompt));
    loop.stop();
}

TEST(PausableThreadLoop, ResumePausedLoopRunsOnce) {
    IterationCounter counter;
    PausableThreadLoop loop([&] { counter.increment(); }, long_sleep);
    loop.resume();
    ASSERT_TRUE(counter.wait_for(1, setup_timeout));
    loop.pause();
    ASSERT_EQ(counter.get(), 1);

    // Resuming a paused loop runs one iteration, after which the loop sleeps as usual.
    EXPECT_TRUE(loop.resume());
    EXPECT_TRUE(loop.is_running());
    ASSERT_TRUE(counter.wait_for(2, prompt));
    EXPECT_FALSE(counter.wait_for(3, std::chrono::milliseconds{100}));

    // Resuming a sleeping loop is not a resume from paused.
    EXPECT_FALSE(loop.resume());
    loop.stop();
}

TEST(PausableThreadLoop, StopInterruptsSleep) {
    IterationCounter counter;
    PausableThreadLoop loop([&] { counter.increment(); }, long_sleep);
    loop.resume();
    ASSERT_TRUE(counter.wait_for(1, setup_timeout));
    std::this_thread::sleep_for(enter_sleep);

    auto const start = std::chrono::steady_clock::now();
    loop.stop();
    EXPECT_LT(std::chrono::steady_clock::now() - start, prompt);
}

TEST(PausableThreadLoop, PauseInterruptsSleep) {
    IterationCounter counter;
    PausableThreadLoop loop([&] { counter.increment(); }, long_sleep);
    loop.resume();
    ASSERT_TRUE(counter.wait_for(1, setup_timeout));
    std::this_thread::sleep_for(enter_sleep);

    auto const start = std::chrono::steady_clock::now();
    loop.pause();
    EXPECT_LT(std::chrono::steady_clock::now() - start, prompt);
    EXPECT_FALSE(loop.is_running());
    loop.stop();
}

TEST(PausableThreadLoop, PauseNbInterruptsSleep) {
    IterationCounter counter;
    PausableThreadLoop loop([&] { counter.increment(); }, long_sleep);
    loop.resume();
    ASSERT_TRUE(counter.wait_for(1, setup_timeout));
    std::this_thread::sleep_for(enter_sleep);

    // pause_nb() returns immediately; the loop must leave its sleep and become paused
    // well before the sleep would have ended.
    auto const start = std::chrono::steady_clock::now();
    loop.pause_nb();
    while (loop.is_running() && std::chrono::steady_clock::now() - start < prompt) {
        std::this_thread::yield();
    }
    EXPECT_LT(std::chrono::steady_clock::now() - start, prompt);
    loop.stop();
}
