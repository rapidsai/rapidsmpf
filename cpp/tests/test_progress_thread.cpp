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

#include <condition_variable>
#include <mutex>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf_test/base_fixture.hpp>

#include <rapidsmp/progress_thread.hpp>

#include "environment.hpp"
#include "rapidsmp/statistics.hpp"

using rapidsmp::FunctionID;
using rapidsmp::ProgressState;
using rapidsmp::ProgressThread;

TEST(ProgressThread, Shutdown) {
    ProgressThread progress_thread(GlobalEnvironment->comm_->logger());

    progress_thread.shutdown();
    EXPECT_THROW(progress_thread.shutdown(), std::logic_error);
}

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
    FunctionID id;
    std::mutex mutex;
    std::condition_variable cv;
};

TEST_P(ProgressThreadEvents, events) {
    size_t const num_threads = std::get<0>(GetParam());
    size_t const num_functions = std::get<1>(GetParam());
    bool const enable_statistics = std::get<2>(GetParam());

    auto& logger = GlobalEnvironment->comm_->logger();
    auto statistics = std::make_shared<rapidsmp::Statistics>(enable_statistics);
    std::vector<std::unique_ptr<ProgressThread>> progress_threads;
    std::vector<std::vector<std::shared_ptr<TestFunction>>> test_functions(num_threads);

    auto expected_count = [num_functions](size_t thread, size_t function) {
        return thread * num_functions + function + 1;
    };

    for (size_t thread = 0; thread < num_threads; ++thread) {
        progress_threads.emplace_back(std::make_unique<ProgressThread>(logger, statistics)
        );

        for (size_t function = 0; function < num_functions; ++function) {
            auto test_function = std::make_shared<TestFunction>();
            auto expected = expected_count(thread, function);
            test_function->id =
                progress_threads[thread]->add_function([test_function, expected]() {
                    ProgressState ret = ProgressState::InProgress;
                    {
                        std::lock_guard<std::mutex> lock(test_function->mutex);
                        if (++test_function->counter == expected) {
                            ret = ProgressState::Done;
                        }
                    }
                    test_function->cv.notify_one();
                    return ret;
                });

            test_functions[thread].emplace_back(test_function);
        }
    }

    for (size_t thread = 0; thread < num_threads; ++thread) {
        for (size_t function = 0; function < num_functions; ++function) {
            auto test_function = test_functions[thread][function];
            auto expected = expected_count(thread, function);
            std::unique_lock<std::mutex> lock(test_function->mutex);
            test_function->cv.wait(lock, [test_function, expected]() {
                return test_function->counter == expected;
            });
            progress_threads[thread]->remove_function(test_function->id);

            EXPECT_EQ(test_function->counter, expected);
        }

        progress_threads[thread]->shutdown();
    }

    if (statistics->enabled() && num_functions > 0) {
        EXPECT_THAT(statistics->report(), ::testing::HasSubstr("event-loop-total"));
    }
}
