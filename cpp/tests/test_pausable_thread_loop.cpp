/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
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

#include <rapidsmpf/pausable_thread_loop.hpp>

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

TEST(PausableThreadLoop, ChaoticOperations) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distr(5, 10);

    PausableThreadLoop loop([&] {
        // thread sleeps for a [5, 10) milliseconds
        std::this_thread::sleep_for(std::chrono::milliseconds(distr(gen)));
    });
    EXPECT_TRUE(loop.resume());

    std::vector<std::future<void>> futures;

    std::atomic<int> stop_count{0};
    for ([[maybe_unused]] auto i : std::views::iota(0, 10)) {
        futures.push_back(std::async(std::launch::async, [&] { loop.pause(); }));
        futures.push_back(std::async(std::launch::async, [&] {
            auto ret = loop.resume();
            if (stop_count > 0) {
                EXPECT_FALSE(ret);
            }
        }));
        futures.push_back(std::async(std::launch::async, [&] { loop.pause_nb(); }));
        futures.push_back(std::async(std::launch::async, [&] {
            auto ret = loop.resume();
            if (stop_count > 0) {
                EXPECT_FALSE(ret);
            }
        }));
        // after this stop call, every other opertion should do nothing
        futures.push_back(std::async(std::launch::async, [&] {
            stop_count += loop.stop();
        }));
    }

    std::ranges::for_each(futures, [](auto& f) { f.get(); });

    EXPECT_FALSE(loop.is_running());
    EXPECT_EQ(1, stop_count);  // loop should be stopped by one of the stop calls
}
