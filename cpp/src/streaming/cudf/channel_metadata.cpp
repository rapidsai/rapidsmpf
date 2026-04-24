/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <stdexcept>
#include <utility>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/memory_reservation.hpp>
#include <rapidsmpf/streaming/cudf/channel_metadata.hpp>

namespace rapidsmpf::streaming {

PartitioningSpec PartitioningSpec::from_order(OrderScheme o) {
    RAPIDSMPF_EXPECTS(
        !o.keys.empty(), "OrderScheme: keys must not be empty", std::invalid_argument
    );
    return {.type = Type::ORDER, .hash = std::nullopt, .order = std::move(o)};
}

bool OrderScheme::operator==(OrderScheme const& other) const {
    if (keys != other.keys || strict_boundaries != other.strict_boundaries) {
        return false;
    }
    bool this_has = boundaries != nullptr;
    bool other_has = other.boundaries != nullptr;
    if (this_has != other_has) {
        return false;
    }
    if (!this_has) {
        return true;
    }
    // Both have boundaries: shape only (see doc in header).
    return boundaries->shape() == other.boundaries->shape();
}

Message to_message(std::uint64_t sequence_number, std::unique_ptr<ChannelMetadata> m) {
    return Message{
        sequence_number,
        std::move(m),
        {},
        [](Message const& msg, MemoryReservation& /*reservation*/) -> Message {
            auto copy = std::make_unique<ChannelMetadata>(msg.get<ChannelMetadata>());
            return Message{msg.sequence_number(), std::move(copy), {}, msg.copy_cb()};
        }
    };
}

}  // namespace rapidsmpf::streaming
