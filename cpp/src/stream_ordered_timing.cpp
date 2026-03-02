/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/stream_ordered_timing.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

namespace {

// Timing holds the state for one in-flight StreamOrderedTiming instance.
//
// Two time points are recorded:
//   host_start            - captured on the CPU when the StreamOrderedTiming object is
//                           constructed.
//   stream_ordered_start  - captured inside the start callback that CUDA executes on the
//                           stream, so it reflects when the GPU actually reached that
//                           point in the stream.
//
// The gap (stream_ordered_start - host_start) is the launch delay: how far ahead the CPU
// ran relative to the GPU.  The gap (stop_cb time - stream_ordered_start) is the stream-
// ordered duration of the enclosed work.
//
// Note, the host_start is always recorded, but is only used when stream_delay_name is
// specified.
struct Timing {
    std::weak_ptr<Statistics> statistics;

    TimePoint stream_ordered_start{};
    std::string stream_ordered_name;

    TimePoint host_start{};
    std::optional<std::string> stream_delay_name;
};

// The global map owns the `Timing` state for all in-flight stream-ordered timers.
//
// We cannot heap-allocate the `void* data` passed to `cudaLaunchHostFunc` and
// free it inside the callback. CUDA does not guarantee that the callback will
// run, for example if the stream is destroyed before execution, so such an
// allocation could leak.
//
// Storing the state in a global map avoids this issue. Entries have a stable
// lifetime independent of whether the callback executes, and `cancel_inflight_timings`
// provides an explicit cleanup path for entries whose callbacks will never run.
std::mutex global_mutex_;
std::uintptr_t global_uid_counter_{0};
std::unordered_map<std::uintptr_t, Timing> global_timings_;

void timing_start_cb(void* data) {
    TimePoint const now = Clock::now();
    std::lock_guard lock(global_mutex_);
    auto const uid = reinterpret_cast<std::uintptr_t>(data);
    auto it = global_timings_.find(uid);
    if (it != global_timings_.end()) {
        it->second.stream_ordered_start = now;
    }
}

void timing_stop_cb(void* data) {
    TimePoint const now = Clock::now();

    std::unique_lock lock(global_mutex_);
    auto const uid = reinterpret_cast<std::uintptr_t>(data);
    auto it = global_timings_.find(uid);
    if (it == global_timings_.end()) {
        return;
    }
    auto timing = global_timings_.extract(it).mapped();
    lock.unlock();

    auto statistics = timing.statistics.lock();
    if (statistics == nullptr) {
        // This can happen if the statistics object has gone out of scope before CUDA gets
        // to this stream operation.
        return;
    }
    statistics->add_stat(
        timing.stream_ordered_name, (now - timing.stream_ordered_start).count()
    );
    if (timing.stream_delay_name) {
        statistics->add_stat(
            *timing.stream_delay_name,
            (timing.stream_ordered_start - timing.host_start).count()
        );
    }
}

}  // namespace

StreamOrderedTiming::StreamOrderedTiming(
    rmm::cuda_stream_view stream, std::shared_ptr<Statistics> statistics
)
    : stream_{stream}, statistics_{std::move(statistics)} {
    RAPIDSMPF_EXPECTS(statistics_ != nullptr, "the statistics pointer cannot be NULL");
    if (!statistics_->enabled()) {
        // Reset to nullptr to make the disabled state permanent for this instance.
        // Otherwise, statistics could be re-enabled between now and stop_and_record().
        statistics_ = nullptr;
        return;
    }
    TimePoint const now = Clock::now();
    std::unique_lock lock(global_mutex_);
    uid_ = global_uid_counter_++;
    global_timings_.insert({uid_, Timing{.statistics = statistics_, .host_start = now}});
    lock.unlock();

    RAPIDSMPF_CUDA_TRY(
        cudaLaunchHostFunc(stream_, timing_start_cb, reinterpret_cast<void*>(uid_))
    );
}

void StreamOrderedTiming::stop_and_record(
    std::string const& name, std::optional<std::string> stream_delay_name
) {
    if (statistics_ == nullptr) {
        return;
    }
    {
        std::lock_guard lock(global_mutex_);
        auto& entry = global_timings_.at(uid_);
        entry.stream_ordered_name = name;
        entry.stream_delay_name = std::move(stream_delay_name);
    }

    RAPIDSMPF_CUDA_TRY(
        cudaLaunchHostFunc(stream_, timing_stop_cb, reinterpret_cast<void*>(uid_))
    );
}

void StreamOrderedTiming::cancel_inflight_timings(Statistics const* statistics) {
    std::lock_guard lock(global_mutex_);

    // Remove all global timings associated with `statistics`, as well as
    // any entries whose weak_ptr has expired.
    std::erase_if(global_timings_, [statistics](auto const& kv) {
        auto const& timing = kv.second;
        auto sp = timing.statistics.lock();
        return !sp || sp.get() == statistics;
    });
}
}  // namespace rapidsmpf
