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
 * @brief A change in reserved-but-not-yet-consumed bytes.
 *
 * Timestamps are derived from a steady-clock offset from a Unix-time anchor, so events
 * in one recording session have non-decreasing timestamps; consecutive events may
 * share a timestamp. Events are appended after the reserved-byte counter changes, so
 * concurrent event order may differ from counter-update order.
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
    std::uint64_t dropped_event_count{0};  ///< Events dropped after append failures.
};

namespace detail {

/**
 * @brief State of an active reservation event recording session.
 *
 * Defined at namespace scope rather than nested in `BufferResource`: clang 20 rejects
 * default-constructing a nested class with default member initializers inside
 * `std::optional` of the enclosing class.
 */
struct RecordingSession {
    std::vector<ReservationEvent> events;  ///< Events recorded since the last drain.
    std::uint64_t dropped_event_count{0};  ///< Events dropped after append failures.
    std::uint64_t id{0};  ///< Session identifier, never zero for an active session.
    std::int64_t unix_time_anchor_ns{0};  ///< `system_clock` time at session start.
    std::int64_t steady_time_anchor_ns{0};  ///< `steady_clock` time at session start.
};

}  // namespace detail

}  // namespace rapidsmpf
