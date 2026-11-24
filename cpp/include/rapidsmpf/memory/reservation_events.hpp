/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

#include <rapidsmpf/memory/memory_type.hpp>

namespace rapidsmpf {

/**
 * @brief Opaque telemetry attribution token carried by memory reservations.
 *
 * Tokens can only be created by `ReservationContextCreator`. A token is attached to a
 * reservation when it is created and is reported unchanged in the `ReservationEvent`s
 * recorded for that reservation.
 *
 * The default-constructed token means unattributed.
 */
class ReservationContext {
    friend class ReservationContextCreator;

  public:
    /// @brief Construct the unattributed context.
    ReservationContext() = default;

    /**
     * @brief Test whether this context carries attribution.
     *
     * @return Whether the context was produced by a `ReservationContextCreator`.
     */
    [[nodiscard]] constexpr bool is_attributed() const noexcept {
        return value_ != 0;
    }

    /**
     * @brief Return a stable hash value for use by language bindings and containers.
     *
     * @return The hash value.
     */
    [[nodiscard]] constexpr std::uint64_t hash() const noexcept {
        return value_;
    }

    /**
     * @brief Compare two contexts for identity.
     *
     * @return Whether both contexts represent the same attribution.
     */
    friend constexpr bool operator==(
        ReservationContext const&, ReservationContext const&
    ) = default;

  private:
    explicit constexpr ReservationContext(std::uint64_t value) : value_{value} {}

    std::uint64_t value_{0};
};

/**
 * @brief Creates globally unique reservation attribution tokens.
 */
class ReservationContextCreator {
  public:
    /**
     * @brief Create a globally unique attributed context.
     *
     * @return A new reservation context.
     */
    [[nodiscard]] ReservationContext create() const noexcept {
        return ReservationContext{next_.fetch_add(1, std::memory_order_relaxed)};
    }

  private:
    inline static std::atomic<std::uint64_t> next_{1};
};

/**
 * @brief A change in reserved, but perhaps not yet used, bytes.
 *
 * A reservation event is created when memory is reserved or released. A
 * positive delta means memory was reserved, while a negative delta means memory
 * was released.
 *
 * Timestamps come from a steady-clock offset from a Unix-time anchor. Events
 * in one recording session are non-decreasing. The actual reservations / releases
 * and these reservation events are not serialized, so the order of recorded
 * events may differ from the order in which the counter was updated when multiple
 * reservation events occur concurrently.
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
    /// incomplete. Release events skipped because their reservation's creation event
    /// was dropped are not counted, so this is a lower bound on the missing events.
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
 *   remember the `id` of the session that recorded their creation, so a release is
 *   only recorded into that same session and never into a later one. A reservation
 *   whose creation event was dropped (cap reached or append failed) remembers no
 *   session, so its release is skipped too. A release is never recorded without
 *   a corresponding reservation, but a reservation might be recorded without a release.
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
 * `dropped_event_count` counts dropped events, not the release events skipped as a
 * result (see Lifecycle), so it is a lower bound on the events missing from the record.
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
