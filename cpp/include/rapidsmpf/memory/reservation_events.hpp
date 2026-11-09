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
 * @brief A change in reserved-but-not-yet-consumed bytes.
 *
 * Timestamps are derived from a steady-clock offset from a Unix-time anchor and are
 * nudged forward when necessary so events in one recording session have strictly
 * increasing timestamps. Events are appended after the reserved-byte counter changes,
 * so concurrent event order may differ from counter-update order.
 */
struct ReservationEvent {
    std::int64_t timestamp_ns;  ///< Strictly increasing Unix timestamp in nanoseconds.
    std::uint64_t reservation_context;  ///< Opaque caller-provided attribution token.
    MemoryType memory_type;  ///< Type of memory whose occupancy changed.
    std::int64_t byte_delta;  ///< Signed change in reserved bytes.
};

/** @brief Events and dropped-event count since the last drain. */
struct ReservationEventBatch {
    std::vector<ReservationEvent> events;  ///< Recorded reservation deltas.
    std::uint64_t dropped_event_count{0};  ///< Events dropped after append failures.
};

}  // namespace rapidsmpf
