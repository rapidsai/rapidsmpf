/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>
#include <stdexcept>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cuda/stream>

#include <rmm/mr/cuda_memory_resource.hpp>

#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils/misc.hpp>

#include "environment.hpp"

using rapidsmpf::ProgressThread;

class ProgressThreadEvents : public ::testing::TestWithParam<std::tuple<int, int, bool>> {
};

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
    std::size_t counter{0};
    ProgressThread::FunctionID id{0, 0};
};

TEST_P(ProgressThreadEvents, events) {
    std::size_t const num_threads = std::get<0>(GetParam());
    std::size_t const num_functions = std::get<1>(GetParam());
    bool const enable_statistics = std::get<2>(GetParam());

    auto statistics = rapidsmpf::Statistics::create(
        enable_statistics ? rapidsmpf::Statistics::Mode::Enabled
                          : rapidsmpf::Statistics::Mode::Disabled
    );
    std::vector<std::unique_ptr<ProgressThread>> progress_threads;
    std::vector<std::vector<std::shared_ptr<TestFunction>>> test_functions(num_threads);

    // The number of times a particular function is expected to be called
    auto expected_count = [num_functions](std::size_t thread, std::size_t function) {
        return thread * num_functions + function + 1;
    };

    for (std::size_t thread = 0; thread < num_threads; ++thread) {
        auto& pt =
            progress_threads.emplace_back(std::make_unique<ProgressThread>(statistics));

        for (std::size_t function = 0; function < num_functions; ++function) {
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

    for (std::size_t thread = 0; thread < num_threads; ++thread) {
        for (std::size_t function = 0; function < num_functions; ++function) {
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
    ProgressThread progress_thread{};

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

TEST(ProgressThreadTests, FunctionErrorIsReportedToCallback) {
    ProgressThread progress_thread;
    std::mutex mutex;
    std::condition_variable cv;
    int num_errors{0};
    std::exception_ptr reported;
    std::atomic<int> num_failing_calls{0};
    std::atomic<int> num_healthy_calls{0};
    std::atomic<bool> healthy_done{false};

    auto failing = progress_thread.add_function(
        [&]() -> ProgressThread::ProgressState {
            ++num_failing_calls;
            throw std::runtime_error("progress function failed");
        },
        [&](std::exception_ptr error) {
            {
                std::lock_guard lock(mutex);
                ++num_errors;
                reported = std::move(error);
            }
            cv.notify_all();
        }
    );
    auto healthy = progress_thread.add_function([&]() {
        ++num_healthy_calls;
        return healthy_done.load() ? ProgressThread::ProgressState::Done
                                   : ProgressThread::ProgressState::InProgress;
    });

    {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return num_errors > 0; });
    }
    EXPECT_THAT(
        [&] { std::rethrow_exception(reported); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::StrEq("progress function failed")
        )
    );

    // Removing a failed function neither blocks nor throws.
    EXPECT_NO_THROW(progress_thread.remove_function(failing));

    // The other functions keep being progressed, and new functions can be added.
    auto const healthy_calls = num_healthy_calls.load();
    while (num_healthy_calls.load() < healthy_calls + 10) {
        std::this_thread::yield();
    }
    auto another =
        progress_thread.add_function([] { return ProgressThread::ProgressState::Done; });
    progress_thread.remove_function(another);
    healthy_done.store(true);
    progress_thread.remove_function(healthy);

    // The failed function was never called again and the error reported only once.
    EXPECT_EQ(num_failing_calls.load(), 1);
    std::lock_guard lock(mutex);
    EXPECT_EQ(num_errors, 1);
}

TEST(ProgressThreadTests, CanDoCudaWorkOnFirstCallback) {
    using namespace rapidsmpf;

    if (!is_pinned_memory_resources_supported()) {
        GTEST_SKIP() << "Pinned memory not supported on this system";
    }

    rmm::mr::cuda_memory_resource cuda_mr;
    auto br = BufferResource::create(cuda_mr, PinnedPoolProperties{});
    auto stream = cuda::stream_ref{cudaStreamLegacy};

    std::exception_ptr eptr;
    ProgressThread progress_thread;
    auto id = progress_thread.add_function([&]() {
        try {
            auto [reservation, _] =
                br->reserve(MemoryType::PINNED_HOST, 1024, AllowOverbooking::YES);
            auto buf = br->make_buffer(1024, stream, reservation);
        } catch (...) {
            eptr = std::current_exception();
        }
        return ProgressThread::ProgressState::Done;
    });
    progress_thread.remove_function(id);

    if (eptr) {
        std::rethrow_exception(eptr);
    }
}
