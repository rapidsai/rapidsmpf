/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>

namespace rapidsmpf {

/// @brief Value types supported by reservation telemetry attributes.
using ReservationAttributeValue = std::variant<std::string, std::int64_t, bool, double>;
/// @brief Map of reservation telemetry attribute names to values.
using ReservationAttributeMap =
    std::unordered_map<std::string, ReservationAttributeValue>;

/**
 * @brief Immutable attributes attached to a reservation request and its reservations.
 *
 * A default value owns no allocation. The pointed-to map is immutable and can be
 * cheaply shared by split and moved reservations.
 */
using ReservationAttributes = std::shared_ptr<ReservationAttributeMap const>;

/**
 * @brief Attributes used by autonomous spill work.
 *
 * @return Shared immutable attributes containing the stable spill site.
 */
inline ReservationAttributes spill_reservation_attributes() {
    static ReservationAttributes const attributes =
        std::make_shared<ReservationAttributeMap const>(
            ReservationAttributeMap{{"rapidsmpf.site", std::string{"spill"}}}
        );
    return attributes;
}

}  // namespace rapidsmpf
