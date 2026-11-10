/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

#include <rapidsmpf/memory/memory_type.hpp>

namespace rapidsmpf {

/**
 * @brief Opaque telemetry attribution token carried by memory reservations.
 *
 * Callers choose the values and rapidsmpf never interprets them: a token is attached to a
 * reservation when it is created and is reported unchanged in the `ReservationEvent`s
 * recorded for that reservation, which lets consumers of the events attribute memory
 * usage to the code that requested it.
 *
 * The value `0` is the default and means unattributed.
 */
using ReservationContext = std::uint64_t;

/**
 * @brief A change in reserved, but perhaps not yet used, bytes.
 *
 * A reservation event is created when memory is reserved or released. A
 * positive delta means memory was reserved, while a negative delta means memory
 * was released.
 *
 */
struct ReservationEvent {
    std::int64_t timestamp_ns;  ///< Non-decreasing Unix timestamp in nanoseconds.
    ReservationContext
        reservation_context;  ///< Opaque caller-provided attribution token.
    MemoryType memory_type;  ///< Type of memory whose occupancy changed.
    std::int64_t byte_delta;  ///< Signed change in reserved bytes.
};

/** @brief Events and dropped-event count since the last drain. */
struct ReservationEventBatch {
    std::vector<ReservationEvent> events;  ///< Recorded reservation deltas.
    /// Events dropped (cap reached or append failed); nonzero means `events` is
    /// incomplete.
    std::uint64_t dropped_event_count{0};
};

namespace detail {

/**
 * @brief State of an active reservation event recording session.
 *
 * @par Thread safety
 * This struct has no synchronization of its own. A `BufferResource` holds at most one
 * instance in `recording_`, and **every** access to it (appending, draining, stopping,
 * and reading `id`) must hold `BufferResource::recording_mutex_`. The lock-free fast
 * path is only the `recording_enabled_` atomic, which lets producers skip the mutex
 * entirely while no session is active. It must be re-checked under the lock, since
 * the session can end in between.
 *
 * @par Lifecycle
 * - `start_reservation_event_recording()` constructs the session, assigns a fresh
 *   non-zero `id` and captures both time anchors.
 * - Producers append one `ReservationEvent` per reservation or release. Reservations
 *   remember the `id` of the session that saw them, so a release is only recorded
 *   into that same session and never into a later one.
 * - `drain_reservation_events()` swaps `events` out and zeroes `dropped_event_count`,
 *   keeping the session (and its `id`) active. Draining is how callers free space
 *   under the `max_reservation_events` cap.
 * - `stop_reservation_event_recording()` takes the remaining events and destroys the
 *   session. After it returns, nothing can append to it.
 *
 * @par Memory bound
 * `events` holds at most `BufferResource::max_reservation_events()` entries. Further
 * events are dropped and counted in `dropped_event_count` until the next drain, so
 * callers that need a complete record should drain more often than the cap fills.
 */
struct RecordingSession {
    std::vector<ReservationEvent> events;  ///< Events recorded since the last drain.
    /// Events dropped (cap reached or append failed) since the last drain.
    std::uint64_t dropped_event_count{0};
    std::uint64_t id{0};  ///< Session identifier, never zero for an active session.
    std::int64_t unix_time_anchor_ns{0};  ///< `system_clock` time at session start.
    std::int64_t steady_time_anchor_ns{0};  ///< `steady_clock` time at session start.
};

}  // namespace detail

}  // namespace rapidsmpf
