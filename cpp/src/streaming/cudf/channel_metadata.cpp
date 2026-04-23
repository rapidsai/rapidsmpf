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

OrderScheme make_order_scheme(
    std::vector<OrderKey> keys,
    std::unique_ptr<TableChunk> boundaries,
    bool strict_boundaries
) {
    RAPIDSMPF_EXPECTS(
        !keys.empty(), "OrderScheme: keys must not be empty", std::invalid_argument
    );
    return OrderScheme{
        .keys = std::move(keys),
        .boundaries = std::move(boundaries),
        .strict_boundaries = strict_boundaries,
    };
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

std::unique_ptr<ChannelMetadata> channel_metadata_from_message(Message msg) {
    return std::make_unique<ChannelMetadata>(msg.release<ChannelMetadata>());
}

ContentDescription get_content_description(ChannelMetadata const& /*m*/) {
    return ContentDescription{ContentDescription::Spillable::NO};
}

Message to_message(std::uint64_t sequence_number, std::unique_ptr<ChannelMetadata> m) {
    auto cd = get_content_description(*m);
    return Message{
        sequence_number,
        std::move(m),
        cd,
        [](Message const& msg, MemoryReservation& /*reservation*/) -> Message {
            auto copy = std::make_unique<ChannelMetadata>(msg.get<ChannelMetadata>());
            auto copy_cd = get_content_description(*copy);
            return Message{
                msg.sequence_number(), std::move(copy), copy_cd, msg.copy_cb()
            };
        }
    };
}

}  // namespace rapidsmpf::streaming
