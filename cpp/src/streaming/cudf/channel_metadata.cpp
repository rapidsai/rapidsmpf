/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <stdexcept>
#include <utility>

#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/streaming/cudf/channel_metadata.hpp>

namespace rapidsmpf::streaming {

PartitioningSpec PartitioningSpec::from_order(OrderScheme o) {
    if (o.keys.empty()) {
        throw std::invalid_argument("OrderScheme: keys must not be empty");
    }
    return {.type = Type::ORDER, .hash = std::nullopt, .order = std::move(o)};
}

bool OrderScheme::operator==(OrderScheme const& other) const {
    if (keys != other.keys || strict_boundary != other.strict_boundary) {
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
    auto tv = boundaries->table_view();
    auto ov = other.boundaries->table_view();
    return tv.num_rows() == ov.num_rows() && tv.num_columns() == ov.num_columns();
}

OrderScheme OrderScheme::clone(MemoryReservation& reservation) const {
    return OrderScheme{
        .keys = keys,
        .boundaries = boundaries
                          ? std::make_unique<TableChunk>(boundaries->copy(reservation))
                          : nullptr,
        .strict_boundary = strict_boundary,
    };
}

PartitioningSpec PartitioningSpec::clone(MemoryReservation& reservation) const {
    switch (type) {
    case Type::NONE:
        return none();
    case Type::INHERIT:
        return inherit();
    case Type::HASH:
        return from_hash(*hash);
    case Type::ORDER:
        return from_order(order->clone(reservation));
    }
    return none();  // unreachable
}

Partitioning Partitioning::clone(MemoryReservation& reservation) const {
    return Partitioning{
        .inter_rank = inter_rank.clone(reservation),
        .local = local.clone(reservation),
    };
}

ChannelMetadata ChannelMetadata::clone(MemoryReservation& reservation) const {
    return ChannelMetadata{local_count, partitioning.clone(reservation), duplicated};
}

ContentDescription content_description_for(ChannelMetadata const& m) {
    ContentDescription cd{ContentDescription::Spillable::NO};
    auto add_spec = [&](PartitioningSpec const& spec) {
        if (spec.type == PartitioningSpec::Type::ORDER && spec.order->boundaries) {
            for (auto mem_type : MEMORY_TYPES) {
                cd.content_size(mem_type) +=
                    spec.order->boundaries->data_alloc_size(mem_type);
            }
        }
    };
    add_spec(m.partitioning.inter_rank);
    add_spec(m.partitioning.local);
    return cd;
}

Message to_message(std::uint64_t sequence_number, std::unique_ptr<ChannelMetadata> m) {
    auto cd = content_description_for(*m);
    return Message{
        sequence_number,
        std::move(m),
        cd,
        [](Message const& msg, MemoryReservation& reservation) -> Message {
            auto copy = std::make_unique<ChannelMetadata>(
                msg.get<ChannelMetadata>().clone(reservation)
            );
            auto copy_cd = content_description_for(*copy);
            return Message{
                msg.sequence_number(), std::move(copy), copy_cd, msg.copy_cb()
            };
        }
    };
}

}  // namespace rapidsmpf::streaming
