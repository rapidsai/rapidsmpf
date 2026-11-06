/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

#include <rapidsmpf/memory/memory_type.hpp>

namespace rapidsmpf {

/** @brief A change in reserved-but-not-yet-consumed bytes. */
struct ReservationEvent {
    std::int64_t timestamp_ns;  ///< Unix timestamp in nanoseconds.
    std::uint64_t reservation_context;  ///< Opaque caller-provided attribution token.
    MemoryType memory_type;  ///< Type of memory whose occupancy changed.
    std::int64_t byte_delta;  ///< Signed change in reserved bytes.
};

/** @brief Events and non-fatal telemetry allocation failures since the last drain. */
struct ReservationEventBatch {
    std::vector<ReservationEvent> events;  ///< Recorded reservation deltas.
    std::uint64_t allocation_failure_count{0};  ///< Failed event-buffer allocations.
};

}  // namespace rapidsmpf
