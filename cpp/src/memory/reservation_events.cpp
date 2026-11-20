/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <utility>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>

namespace rapidsmpf {

namespace {

template <typename Clock>
std::int64_t clock_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               Clock::now().time_since_epoch()
    )
        .count();
}

}  // namespace

std::uint64_t BufferResource::record_reservation_event(
    ReservationContext reservation_context,
    MemoryType memory_type,
    std::int64_t byte_delta,
    std::uint64_t expected_session
) noexcept {
    if (!recording_enabled_.load(std::memory_order_acquire)) {
        return 0;
    }
    std::lock_guard lock(recording_mutex_);
    if (!recording_ || (expected_session != 0 && recording_->id != expected_session)) {
        return 0;
    }
    // Returning zero for a dropped event tells the caller the session did not observe
    // it. A reservation stores this as `observed_session_`, and `release()` only
    // records a release event for reservations whose creation was observed.
    if (recording_->events.size() >= max_reservation_events_) {
        ++recording_->dropped_event_count;
        return 0;
    }
    try {
        auto const elapsed =
            clock_now_ns<std::chrono::steady_clock>() - recording_->steady_time_anchor_ns;
        recording_->events.push_back(
            {recording_->unix_time_anchor_ns + elapsed,
             reservation_context,
             memory_type,
             byte_delta}
        );
    } catch (...) {
        // Allocation failure while appending, dropped like a capped event (see above).
        ++recording_->dropped_event_count;
        return 0;
    }
    return recording_->id;
}

void BufferResource::start_reservation_event_recording() {
    std::lock_guard lock(recording_mutex_);
    RAPIDSMPF_EXPECTS(
        !recording_.has_value(),
        "reservation event recording is already active",
        std::logic_error
    );
    recording_.emplace();
    recording_->id = next_recording_session_id_++;
    recording_->unix_time_anchor_ns = clock_now_ns<std::chrono::system_clock>();
    recording_->steady_time_anchor_ns = clock_now_ns<std::chrono::steady_clock>();
    recording_enabled_.store(true, std::memory_order_release);
}

ReservationEventBatch BufferResource::drain_reservation_events() {
    ReservationEventBatch result;
    std::lock_guard lock(recording_mutex_);
    if (recording_) {
        result.events.swap(recording_->events);
        result.dropped_event_count = std::exchange(recording_->dropped_event_count, 0);
    }
    return result;
}

ReservationEventBatch BufferResource::stop_reservation_event_recording() {
    ReservationEventBatch result;
    std::lock_guard lock(recording_mutex_);
    recording_enabled_.store(false, std::memory_order_release);
    if (recording_) {
        result.events.swap(recording_->events);
        result.dropped_event_count = recording_->dropped_event_count;
        recording_.reset();
    }
    return result;
}

}  // namespace rapidsmpf
