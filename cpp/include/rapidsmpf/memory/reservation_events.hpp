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

#include <rapidsmpf/memory/reservation_attributes.hpp>

namespace rapidsmpf {

/** @brief Memory type encoded in reservation telemetry. */
enum class ReservationEventMemoryType : std::uint32_t {
    DEVICE = 0,
    HOST = 1,
    PINNED_HOST = 2,
    DISK = 3
};

/** @brief Policy used by a reservation request. */
enum class ReservationRequestPolicy : std::uint32_t {
    IMMEDIATE = 0,
    OVERBOOK = 1,
    SPILL = 2,
    WAIT = 3,
    WAIT_OR_OVERBOOK = 4,
    WAIT_OR_FAIL = 5
};

/** @brief Terminal outcome of a reservation request. */
enum class ReservationRequestOutcome : std::uint32_t {
    GRANTED = 0,
    EMPTY = 1,
    FAILED = 2,
    CANCELLED = 3
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
    std::vector<ReservationEventMemoryType> memory_types;  ///< Memory preference order.
    ReservationRequestPolicy policy;  ///< Request policy.
    std::optional<std::int64_t> net_memory_delta;  ///< Optional expected net change.
    ReservationAttributeMap attributes;  ///< Owned request attributes.
};

/** @brief Payload emitted when a reservation request begins waiting. */
struct RequestWaitingEvent {
    std::uint64_t id;  ///< Request identifier.
};

/** @brief Payload emitted when a reservation request terminates. */
struct RequestResolvedEvent {
    std::uint64_t id;  ///< Request identifier.
    ReservationRequestOutcome outcome;  ///< Terminal outcome.
    std::uint64_t reservation_id;  ///< Granted reservation identifier, or zero.
    ReservationEventMemoryType memory_type;  ///< Granted memory type.
    std::size_t size;  ///< Granted bytes.
    std::size_t overbooked_bytes;  ///< Bytes granted beyond availability.
    bool timed_out;  ///< Whether forced progress resolved the request.
    std::string error;  ///< Failure description.
};

/** @brief Payload emitted when a reservation is created. */
struct ReservationCreatedEvent {
    std::uint64_t id;  ///< Reservation identifier.
    ReservationEventMemoryType memory_type;  ///< Reserved memory type.
    std::size_t size;  ///< Initial reserved bytes.
    ReservationAttributeMap attributes;  ///< Owned reservation attributes.
    std::uint64_t request_id;  ///< Originating request identifier, or zero.
    std::uint64_t parent_id;  ///< Parent reservation identifier, or zero.
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
    ReservationEventMemoryType memory_type;  ///< Changed memory type.
    std::int64_t configured_limit;  ///< New signed configured limit.
};

/** @brief Variant containing every reservation event payload. */
using ReservationEventData = std::variant<
    RequestSubmittedEvent,
    RequestWaitingEvent,
    RequestResolvedEvent,
    ReservationCreatedEvent,
    ReservationResizedEvent,
    ReservationReleasedEvent,
    MemoryLimitChangedEvent>;

/**
 * @brief One owned reservation event.
 *
 * `sequence` starts at zero for each recording session. `timestamp_ns` is the
 * event occurrence time as Unix nanoseconds. Payload strings, vectors, and
 * attributes are owned by the event and remain valid after the producer returns.
 */
struct ReservationEvent {
    std::uint64_t sequence;  ///< Global sequence within the recording session.
    std::int64_t timestamp_ns;  ///< Unix timestamp in nanoseconds.
    ReservationEventData data;  ///< Owned event payload.
};

/** @brief An ordered batch removed from the in-memory event buffer. */
using ReservationEventBatch = std::vector<ReservationEvent>;

}  // namespace rapidsmpf
