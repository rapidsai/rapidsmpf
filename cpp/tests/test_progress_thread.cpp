/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <future>

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

TEST(ProgressThreadTests, BoundedTransferEventRecorder) {
    using namespace rapidsmpf;

    ProgressThread progress_thread;
    progress_thread.enable_transfer_events(2);
    progress_thread.record_transfer_event(
        42, CollectiveKind::ALLGATHER, 1, 2, 7, 11, 13, MemoryType::PINNED_HOST
    );
    // Self-transfers and empty control messages are not data-channel events.
    progress_thread.record_transfer_event(
        42, CollectiveKind::SHUFFLER, 2, 2, 8, 1, 1, MemoryType::DEVICE
    );
    progress_thread.record_transfer_event(
        42, CollectiveKind::SHUFFLER, 1, 2, 9, 0, 0, MemoryType::DEVICE
    );
    progress_thread.record_transfer_event(
        43, CollectiveKind::SPARSE_ALLTOALL, 3, 2, 10, 17, 19, MemoryType::HOST
    );
    progress_thread.record_transfer_event(
        44, CollectiveKind::ALLREDUCE, 4, 2, 11, 0, 23, MemoryType::DEVICE
    );

    EXPECT_EQ(progress_thread.dropped_transfer_events(), 1);
    auto events = progress_thread.drain_transfer_events();
    ASSERT_EQ(events.size(), 2);
    EXPECT_EQ(events[0].op_id, 42);
    EXPECT_EQ(events[0].collective_kind, CollectiveKind::ALLGATHER);
    EXPECT_EQ(events[0].source_rank, 1);
    EXPECT_EQ(events[0].destination_rank, 2);
    EXPECT_EQ(events[0].message_id, 7);
    EXPECT_EQ(events[0].metadata_bytes, 11);
    EXPECT_EQ(events[0].payload_bytes, 13);
    EXPECT_EQ(events[0].destination_memory_type, MemoryType::PINNED_HOST);
    EXPECT_GT(events[0].completion_timestamp_ns, 0);
    EXPECT_LE(events[0].completion_timestamp_ns, events[1].completion_timestamp_ns);
    EXPECT_TRUE(progress_thread.drain_transfer_events().empty());

    progress_thread.disable_transfer_events();
    progress_thread.record_transfer_event(
        45, CollectiveKind::ALLREDUCE, 1, 2, 12, 0, 29, MemoryType::DEVICE
    );
    EXPECT_TRUE(progress_thread.drain_transfer_events().empty());
}
