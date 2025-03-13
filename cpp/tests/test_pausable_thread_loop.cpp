/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <atomic>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmp/pausable_thread_loop.hpp>

using rapidsmp::detail::PausableThreadLoop;

// Test if the pause functionality works
TEST(PausableThreadLoop, ResumeAndPause) {
    std::atomic<int> counter{0};
    PausableThreadLoop loop([&]() { counter.fetch_add(1, std::memory_order_relaxed); });

    // The loop starts paused.
    EXPECT_TRUE(!loop.is_running());
    loop.resume();
    EXPECT_TRUE(loop.is_running());

    // Let the loop run and check counter has been increased.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_GT(counter.load(std::memory_order_relaxed), 0);

    // If we pause, the counter should stay the same.
    loop.pause();
    int count_after_pause = counter.load(std::memory_order_relaxed);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    loop.stop();
    EXPECT_EQ(counter.load(std::memory_order_relaxed), count_after_pause);
}
