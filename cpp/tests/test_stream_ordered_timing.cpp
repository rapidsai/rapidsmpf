/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <future>

#include <gtest/gtest.h>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/cuda_event.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/stream_ordered_timing.hpp>

#include "utils.hpp"

using namespace rapidsmpf;

TEST(StreamOrderedTiming, Disabled) {
    // With disabled statistics, stop_and_record is a no-op: no stat is written.
    rmm::cuda_stream stream;
    auto stats = Statistics::disabled();
    StreamOrderedTiming timing{stream.view(), stats};
    timing.stop_and_record("test");
    stream.synchronize();
    EXPECT_THROW(stats->get_stat("test"), std::out_of_range);
}

TEST(StreamOrderedTiming, RecordsDuration) {
    // A positive duration is recorded under the given name after stream sync.
    rmm::cuda_stream stream;
    auto stats = std::make_shared<Statistics>();
    {
        StreamOrderedTiming timing{stream.view(), stats};
        // Do a small GPU operation so there is measurable work between start and stop.
        rmm::device_buffer buf(1_MiB, stream.view());
        RAPIDSMPF_CUDA_TRY(cudaMemsetAsync(buf.data(), 0, 1_MiB, stream.value()));
        timing.stop_and_record("my-timing");
    }
    stream.synchronize();
    auto const stat = stats->get_stat("my-timing");
    EXPECT_EQ(stat.count(), 1);
    EXPECT_GT(stat.value(), 0.0);
}

TEST(StreamOrderedTiming, MultipleTimings) {
    // Repeated timings for the same name accumulate: count equals the number of timings.
    rmm::cuda_stream stream;
    auto stats = std::make_shared<Statistics>();
    constexpr int n = 3;
    for (int i = 0; i < n; ++i) {
        StreamOrderedTiming timing{stream.view(), stats};
        timing.stop_and_record("my-timing");
    }
    stream.synchronize();
    auto const stat = stats->get_stat("my-timing");
    EXPECT_EQ(stat.count(), static_cast<std::size_t>(n));
    EXPECT_GT(stat.value(), 0.0);
}

TEST(StreamOrderedTiming, Cancel) {
    // Use a CPU-side gate backed by std::promise/future so that no callbacks can
    // execute until after cancel_inflight_timings is called. This guarantees that
    // cancelled timings are not recorded, while timings for other Statistics objects
    // are still recorded. Calling cancel when no timings are pending is a no-op.
    rmm::cuda_stream stream;
    auto stats_a = std::make_shared<Statistics>();
    auto stats_b = std::make_shared<Statistics>();

    // Block stream: no callbacks will execute until gate.open() is called.
    struct Gate {
        std::promise<void> promise{};
        std::future<void> future{promise.get_future()};

        static void wait_cb(void* data) noexcept {
            static_cast<Gate*>(data)->future.wait();
        }

        void open() {
            promise.set_value();
        }
    };

    Gate gate;
    RAPIDSMPF_CUDA_TRY(cudaLaunchHostFunc(stream.view(), Gate::wait_cb, &gate));
    {
        StreamOrderedTiming timing_a{stream.view(), stats_a};
        timing_a.stop_and_record("timing-a");
        StreamOrderedTiming timing_b{stream.view(), stats_b};
        timing_b.stop_and_record("timing-b");
    }

    // Cancel while the stream is still blocked.
    StreamOrderedTiming::cancel_inflight_timings(stats_a.get());

    // Unblock the stream and let all remaining callbacks run.
    gate.open();
    stream.synchronize();

    EXPECT_THROW(stats_a->get_stat("timing-a"), std::out_of_range);
    EXPECT_EQ(stats_b->get_stat("timing-b").count(), 1);

    // Calling again when no timings are pending is safe.
    EXPECT_NO_THROW(StreamOrderedTiming::cancel_inflight_timings(stats_a.get()));
}

TEST(StreamOrderedTiming, StreamDelay) {
    rmm::cuda_stream stream;
    auto stats = std::make_shared<Statistics>();

    // With a stream_delay_name, the delay stat is recorded and is >= 0.
    {
        StreamOrderedTiming timing{stream.view(), stats};
        timing.stop_and_record("my-timing", "my-stream-delay");
    }
    stream.synchronize();
    auto const delay = stats->get_stat("my-stream-delay");
    EXPECT_EQ(delay.count(), 1);
    EXPECT_GE(delay.value(), 0.0);

    // With std::nullopt (the default), no stream-delay stat is written.
    auto stats2 = std::make_shared<Statistics>();
    {
        StreamOrderedTiming timing{stream.view(), stats2};
        timing.stop_and_record("my-timing");
    }
    stream.synchronize();
    EXPECT_THROW(stats2->get_stat("my-stream-delay"), std::out_of_range);
}

TEST(StreamOrderedTiming, StatisticsDestroyedBeforeStreamSync) {
    // If the Statistics object is destroyed before the stream callbacks execute,
    // the callbacks detect the expired weak_ptr and return without crashing.
    rmm::cuda_stream stream;
    {
        auto stats = std::make_shared<Statistics>();
        StreamOrderedTiming timing{stream.view(), stats};
        timing.stop_and_record("my-timing");
        // Both `timing` and `stats` go out of scope here. The shared_ptr count
        // drops to zero, destroying Statistics before the stream callbacks run.
    }
    EXPECT_NO_THROW(stream.synchronize());
}
