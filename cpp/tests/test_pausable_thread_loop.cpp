/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <random>
#include <ranges>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cuda/stream>

#include <rmm/mr/cuda_memory_resource.hpp>

#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/pausable_thread_loop.hpp>
#include <rapidsmpf/utils/misc.hpp>

#include "pausable_thread_loop_internal.hpp"

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

namespace {

using LoopState = rapidsmpf::detail::LoopState;
using State = LoopState::State;
using Step = LoopState::Step;

std::exception_ptr make_error() {
    return std::make_exception_ptr(std::runtime_error("loop function failed"));
}

// Once a failure is observable, the loop is not running and cannot be paused, resumed.
void expect_failure_is_terminal(LoopState& s) {
    EXPECT_NE(s.error(), nullptr);
    EXPECT_FALSE(s.is_running());
    EXPECT_FALSE(s.resume());
    EXPECT_FALSE(s.pause_nb());
    EXPECT_FALSE(s.is_running());
}

}  // namespace

// The `LoopState` tests mostly execute operations sequentially. Since every operation
// runs under a single mutex, this covers the possible interleavings of concurrent
// operations.

TEST(LoopState, PauseResumeStop) {
    LoopState s;
    EXPECT_EQ(s.state(), State::Paused);
    EXPECT_FALSE(s.is_running());
    EXPECT_TRUE(s.resume());
    EXPECT_EQ(s.next_step(), Step::Run);
    EXPECT_TRUE(s.pause_nb());
    EXPECT_EQ(s.state(), State::Pausing);
    {
        // The loop thread acknowledges the pause, then waits until resumed.
        std::thread loop_thread([&] { EXPECT_EQ(s.next_step(), Step::Run); });
        while (s.state() != State::Paused) {
            std::this_thread::yield();
        }
        EXPECT_FALSE(s.is_running());
        EXPECT_TRUE(s.resume());
        loop_thread.join();
    }
    EXPECT_TRUE(s.stop());
    EXPECT_EQ(s.state(), State::Stopping);
    EXPECT_TRUE(s.is_running());
    EXPECT_EQ(s.next_step(), Step::Exit);
    EXPECT_EQ(s.state(), State::Stopped);
    EXPECT_FALSE(s.stop());
    EXPECT_FALSE(s.resume());
    EXPECT_EQ(s.error(), nullptr);
}

TEST(LoopState, FailWhileRunning) {
    LoopState s;
    s.resume();
    s.fail(make_error());
    EXPECT_EQ(s.state(), State::Failed);
    expect_failure_is_terminal(s);
    EXPECT_TRUE(s.stop());
    EXPECT_EQ(s.state(), State::Stopped);
    EXPECT_EQ(s.next_step(), Step::Exit);
    expect_failure_is_terminal(s);
    EXPECT_FALSE(s.stop());
}

TEST(LoopState, FailWhilePausing) {
    LoopState s;
    s.resume();
    s.pause_nb();
    s.fail(make_error());
    // Not left PAUSING, so `pause()` doesn't block.
    ASSERT_EQ(s.state(), State::Failed);
    s.pause();
    expect_failure_is_terminal(s);
}

TEST(LoopState, FailWhileStopping) {
    LoopState s;
    s.resume();
    EXPECT_TRUE(s.stop());
    s.fail(make_error());
    // The failure completes the stop.
    EXPECT_EQ(s.state(), State::Stopped);
    expect_failure_is_terminal(s);
    EXPECT_EQ(s.next_step(), Step::Exit);
    EXPECT_FALSE(s.stop());
}

TEST(PausableThreadLoop, ExceptionIsReported) {
    std::atomic<int> num_calls{0};
    std::promise<std::exception_ptr> reported;
    PausableThreadLoop loop(
        [&]() {
            ++num_calls;
            throw std::runtime_error("loop function failed");
        },
        std::chrono::seconds{0},
        [&](std::exception_ptr error) {
            // The failure is already observable and the loop not running.
            EXPECT_EQ(loop.error(), error);
            EXPECT_FALSE(loop.is_running());
            reported.set_value(std::move(error));
        }
    );
    EXPECT_TRUE(loop.resume());
    auto error = reported.get_future().get();
    EXPECT_THROW(std::rethrow_exception(error), std::runtime_error);
    EXPECT_FALSE(loop.resume());
    loop.pause();  // Doesn't block.
    EXPECT_TRUE(loop.stop());
    EXPECT_EQ(loop.error(), error);
    EXPECT_EQ(num_calls.load(), 1);
}

TEST(PausableThreadLoop, StopFromErrorHandler) {
    std::promise<bool> stopped;
    PausableThreadLoop loop(
        []() { throw std::runtime_error("loop function failed"); },
        std::chrono::seconds{0},
        [&](std::exception_ptr) { stopped.set_value(loop.stop()); }
    );
    loop.resume();
    EXPECT_TRUE(stopped.get_future().get());
    EXPECT_FALSE(loop.is_running());
    EXPECT_FALSE(loop.stop());
    // The destructor joins the thread.
}

TEST(PausableThreadLoop, StopFromLoopFunction) {
    // The thread exits without the sleep between iterations, which would otherwise delay
    // the destructor joining it by `sleep`.
    constexpr auto sleep = std::chrono::seconds{30};
    std::atomic<int> num_calls{0};
    auto const start = std::chrono::steady_clock::now();
    {
        std::promise<bool> stopped;
        PausableThreadLoop loop(
            [&]() {
                if (++num_calls == 1) {
                    stopped.set_value(loop.stop());
                }
            },
            sleep
        );
        loop.resume();
        EXPECT_TRUE(stopped.get_future().get());
        EXPECT_FALSE(loop.stop());
        // The destructor joins the thread.
    }
    EXPECT_LT(std::chrono::steady_clock::now() - start, sleep);
    EXPECT_EQ(num_calls.load(), 1);
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
