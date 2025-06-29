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

#include <future>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf_test/base_fixture.hpp>

#include <rapidsmpf/progress_thread.hpp>

#include "environment.hpp"
#include "rapidsmpf/statistics.hpp"

using rapidsmpf::ProgressThread;

class ProgressThreadEvents
    : public cudf::test::BaseFixtureWithParam<std::tuple<int, int, bool>> {};

// test different `num_threads` and `num_functions`.
INSTANTIATE_TEST_SUITE_P(
    ProgressThread,
    ProgressThreadEvents,
    testing::Combine(
        testing::Values(1, 2, 4, 8),  // num_threads
        testing::Values(0, 1, 2, 4, 8),  // num_functions
        testing::Values(false, true)  // enable_statistics
    )
);

struct TestFunction {
    size_t counter{0};
    ProgressThread::FunctionID id{0, 0};
};

TEST_P(ProgressThreadEvents, events) {
    size_t const num_threads = std::get<0>(GetParam());
    size_t const num_functions = std::get<1>(GetParam());
    bool const enable_statistics = std::get<2>(GetParam());

    auto& logger = GlobalEnvironment->comm_->logger();
    auto statistics = std::make_shared<rapidsmpf::Statistics>(enable_statistics);
    std::vector<std::unique_ptr<ProgressThread>> progress_threads;
    std::vector<std::vector<std::shared_ptr<TestFunction>>> test_functions(num_threads);

    // The number of times a particular function is expected to be called
    auto expected_count = [num_functions](size_t thread, size_t function) {
        return thread * num_functions + function + 1;
    };

    for (size_t thread = 0; thread < num_threads; ++thread) {
        auto& pt = progress_threads.emplace_back(
            std::make_unique<ProgressThread>(logger, statistics)
        );

        for (size_t function = 0; function < num_functions; ++function) {
            auto test_function = std::make_shared<TestFunction>();
            auto expected = expected_count(thread, function);

            test_function->id = pt->add_function([test_function, expected]() {
                if (++test_function->counter == expected) {
                    return ProgressThread::ProgressState::Done;
                } else {
                    return ProgressThread::ProgressState::InProgress;
                }
            });

            test_functions[thread].push_back(std::move(test_function));
        }
    }

    for (size_t thread = 0; thread < num_threads; ++thread) {
        for (size_t function = 0; function < num_functions; ++function) {
            auto test_function = test_functions[thread][function];
            progress_threads[thread]->remove_function(test_function->id);
            EXPECT_EQ(test_function->counter, expected_count(thread, function));
        }

        progress_threads[thread]->stop();
    }

    if (statistics->enabled() && num_functions > 0) {
        EXPECT_THAT(statistics->report(), ::testing::HasSubstr("event-loop-total"));
    }
}

TEST(ProgressThreadTests, RemoveFunctionWithDelayedPause) {
    ProgressThread progress_thread(GlobalEnvironment->comm_->logger());

    // add a function to the progress thread that never completes
    auto id = progress_thread.add_function([] {
        return ProgressThread::ProgressState::InProgress;
    });

    // pause the progress thread asynchronously after a short delay
    auto future = std::async(std::launch::async, [&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        progress_thread.pause();
        EXPECT_FALSE(progress_thread.is_running());
    });

    // attempt to remove the function. This will block until the progress thread is
    // paused, because the function will never complete.
    progress_thread.remove_function(id);

    future.get();
}
