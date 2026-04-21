/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <stdexcept>
#include <utility>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
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
    bool strict_boundary
) {
    RAPIDSMPF_EXPECTS(
        !keys.empty(), "OrderScheme: keys must not be empty", std::invalid_argument
    );
    return OrderScheme{
        .keys = std::move(keys),
        .boundaries = std::move(boundaries),
        .strict_boundary = strict_boundary,
    };
}

void partitioning_spec_set_none(PartitioningSpec& spec) {
    spec = PartitioningSpec::none();
}

void partitioning_spec_set_inherit(PartitioningSpec& spec) {
    spec = PartitioningSpec::inherit();
}

void partitioning_spec_set_hash(PartitioningSpec& spec, HashScheme hash_scheme) {
    spec = PartitioningSpec::from_hash(std::move(hash_scheme));
}

void partitioning_spec_set_order(PartitioningSpec& spec, OrderScheme& src) {
    OrderScheme o{
        .keys = src.keys,
        .boundaries = std::move(src.boundaries),
        .strict_boundary = src.strict_boundary,
    };
    spec = PartitioningSpec::from_order(std::move(o));
}

OrderScheme* partitioning_spec_order_scheme_ptr(PartitioningSpec& spec) {
    RAPIDSMPF_EXPECTS(
        spec.type == PartitioningSpec::Type::ORDER,
        "partitioning_spec_order_scheme_ptr: spec must be ORDER",
        std::logic_error
    );
    return &spec.order.value();
}

cudf::size_type order_scheme_boundary_row_count(OrderScheme const* scheme) {
    RAPIDSMPF_EXPECTS(
        scheme != nullptr && scheme->boundaries != nullptr,
        "order_scheme_boundary_row_count: boundaries must be set",
        std::logic_error
    );
    return scheme->boundaries->shape().first;
}

cudf::table_view order_scheme_boundaries_table_view(OrderScheme const* scheme) {
    RAPIDSMPF_EXPECTS(
        scheme != nullptr && scheme->boundaries != nullptr
            && scheme->boundaries->is_available(),
        "ORDER boundaries must be device-resident (metadata does not unspill them)",
        std::runtime_error
    );
    return scheme->boundaries->table_view();
}

cudaStream_t order_scheme_boundaries_cuda_stream(OrderScheme const* scheme) {
    RAPIDSMPF_EXPECTS(
        scheme != nullptr && scheme->boundaries != nullptr
            && scheme->boundaries->is_available(),
        "ORDER boundaries must be device-resident (metadata does not unspill them)",
        std::runtime_error
    );
    return scheme->boundaries->stream().value();
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
    return boundaries->shape() == other.boundaries->shape();
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

std::unique_ptr<ChannelMetadata> make_channel_metadata(
    std::uint64_t local_count, Partitioning& partitioning, bool duplicated
) {
    auto clone_spec = [](PartitioningSpec& spec) -> PartitioningSpec {
        using T = PartitioningSpec::Type;
        switch (spec.type) {
        case T::NONE:
            return PartitioningSpec::none();
        case T::INHERIT:
            return PartitioningSpec::inherit();
        case T::HASH:
            return PartitioningSpec::from_hash(*spec.hash);
        case T::ORDER:
            {
                OrderScheme o{
                    .keys = spec.order->keys,
                    .boundaries = std::move(spec.order->boundaries),
                    .strict_boundary = spec.order->strict_boundary,
                };
                return PartitioningSpec::from_order(std::move(o));
            }
        }
        return PartitioningSpec::none();
    };
    Partitioning part{
        .inter_rank = clone_spec(partitioning.inter_rank),
        .local = clone_spec(partitioning.local),
    };
    return std::make_unique<ChannelMetadata>(local_count, std::move(part), duplicated);
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
        [](Message const& msg, MemoryReservation& reservation) -> Message {
            auto copy = std::make_unique<ChannelMetadata>(
                msg.get<ChannelMetadata>().clone(reservation)
            );
            auto copy_cd = get_content_description(*copy);
            return Message{
                msg.sequence_number(), std::move(copy), copy_cd, msg.copy_cb()
            };
        }
    };
}

}  // namespace rapidsmpf::streaming
