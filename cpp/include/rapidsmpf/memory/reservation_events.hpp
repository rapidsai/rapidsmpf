/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/memory/reservation_attributes.hpp>
#include <rapidsmpf/utils/time.hpp>

namespace rapidsmpf {

/** @brief Policy used by a reservation request. */
enum class ReservationRequestPolicy : std::uint32_t {
    IMMEDIATE = 0,
    OVERBOOK = 1,
    SPILL = 2,
    WAIT = 3,
    WAIT_OR_OVERBOOK = 4,
    WAIT_OR_FAIL = 5
};

/** @brief Reason an active reservation changed size. */
enum class ReservationResizeReason : std::uint32_t {
    CONSUMED = 0,
    SPLIT_OFF = 1
};

/** @brief Reason a reservation reached its terminal state. */
enum class ReservationReleaseReason : std::uint32_t {
    CLEARED = 0,
    RELEASED_CONSUMED = 1,
    RELEASED_SPLIT = 2
};

/** @brief Payload emitted when a reservation request starts. */
struct RequestSubmittedEvent {
    std::uint64_t id;  ///< Request identifier.
    std::size_t size;  ///< Requested bytes.
    std::vector<MemoryType> memory_types;  ///< Memory preference order.
    ReservationRequestPolicy policy;  ///< Request policy.
    std::optional<std::int64_t> net_memory_delta;  ///< Optional expected net change.
    ReservationAttributes attributes;  ///< Shared immutable request attributes.
};

/** @brief Payload emitted when a reservation request begins waiting. */
struct RequestWaitingEvent {
    std::uint64_t id;  ///< Request identifier.
};

/** @brief Payload emitted when a reservation request is granted. */
struct RequestGrantedEvent {
    std::uint64_t id;  ///< Request identifier.
    /// Granted reservation identifier, absent when the reservation is not recorded,
    /// e.g. a zero-size reservation.
    std::optional<std::uint64_t> reservation_id;
    MemoryType memory_type;  ///< Granted memory type.
    std::size_t size;  ///< Granted bytes.
    std::size_t overbooked_bytes;  ///< Bytes granted beyond availability.
    bool timed_out;  ///< Whether the grant was forced after the reserve timeout.
};

/** @brief Payload emitted when a reservation request ends with nothing reserved. */
struct RequestEmptyEvent {
    std::uint64_t id;  ///< Request identifier.
    bool timed_out;  ///< Whether forced progress after the reserve timeout ended it.
};

/** @brief Payload emitted when a reservation request fails with an error. */
struct RequestFailedEvent {
    std::uint64_t id;  ///< Request identifier.
    std::string error;  ///< Failure description.
};

/** @brief Payload emitted when a waiting reservation request is cancelled. */
struct RequestCancelledEvent {
    std::uint64_t id;  ///< Request identifier.
};

/**
 * @brief Payload emitted when a reservation is created.
 *
 * Exactly one of `request_id` and `parent_id` is set: a reservation created by a
 * request has a `request_id`, and a reservation split off another one has a
 * `parent_id`.
 */
struct ReservationCreatedEvent {
    std::uint64_t id;  ///< Reservation identifier.
    MemoryType memory_type;  ///< Reserved memory type.
    std::size_t size;  ///< Initial reserved bytes.
    ReservationAttributes attributes;  ///< Shared immutable reservation attributes.
    std::optional<std::uint64_t> request_id;  ///< Originating request identifier.
    std::optional<std::uint64_t> parent_id;  ///< Parent reservation identifier.
};

/** @brief Payload emitted when an active reservation changes size. */
struct ReservationResizedEvent {
    std::uint64_t id;  ///< Reservation identifier.
    std::size_t size;  ///< Remaining reserved bytes.
    ReservationResizeReason reason;  ///< Cause of the size change.
};

/** @brief Payload emitted when a reservation terminates. */
struct ReservationReleasedEvent {
    std::uint64_t id;  ///< Reservation identifier.
    ReservationReleaseReason reason;  ///< Cause of release.
};

/** @brief Payload emitted when a configured memory limit changes. */
struct MemoryLimitChangedEvent {
    MemoryType memory_type;  ///< Changed memory type.
    std::int64_t configured_limit;  ///< New signed configured limit.
};

/** @brief Variant containing every reservation event payload. */
using ReservationEventData = std::variant<
    RequestSubmittedEvent,
    RequestWaitingEvent,
    RequestGrantedEvent,
    RequestEmptyEvent,
    RequestFailedEvent,
    RequestCancelledEvent,
    ReservationCreatedEvent,
    ReservationResizedEvent,
    ReservationReleasedEvent,
    MemoryLimitChangedEvent>;

/**
 * @brief One reservation event.
 *
 * `sequence` starts at zero for each recording session. `steady_time_ns` is the
 * event occurrence time read from `event_time_ns()`. Event times are
 * nondecreasing in `sequence` order; consecutive events may share a time.
 *
 * Every event occurrence takes the next `sequence`, including events dropped
 * because the buffer reached `BufferResource::reservation_event_limit()`. A gap in
 * `sequence` therefore marks dropped events.
 */
struct ReservationEvent {
    std::uint64_t sequence;  ///< Global sequence within the recording session.
    std::int64_t steady_time_ns;  ///< Steady-clock time in nanoseconds.
    ReservationEventData data;  ///< Event payload.
};

/** @brief An ordered batch removed from the in-memory event buffer. */
using ReservationEventBatch = std::vector<ReservationEvent>;

}  // namespace rapidsmpf
