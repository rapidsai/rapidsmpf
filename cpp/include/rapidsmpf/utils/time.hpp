/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <chrono>
#include <cstdint>

namespace rapidsmpf {

/**
 * @brief Current time on the clock used to timestamp recorded events.
 *
 * Reads `std::chrono::steady_clock`, which is monotonic and has an unspecified
 * epoch. To convert event times to another time domain, such as Unix time, sample
 * this function and the target clock at about the same time and add the difference
 * to each event time.
 *
 * @return Nanoseconds since the steady clock's epoch.
 */
[[nodiscard]] inline std::int64_t event_time_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()
    )
        .count();
}

}  // namespace rapidsmpf
