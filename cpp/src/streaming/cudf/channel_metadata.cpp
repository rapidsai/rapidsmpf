/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <stdexcept>
#include <utility>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/memory_reservation.hpp>
#include <rapidsmpf/streaming/cudf/channel_metadata.hpp>

namespace rapidsmpf::streaming {

OrderScheme::OrderScheme(
    std::vector<OrderKey> keys,
    std::shared_ptr<TableChunk> boundaries,
    bool strict_boundaries
) {
    RAPIDSMPF_EXPECTS(
        !keys.empty(), "OrderScheme: keys must not be empty", std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        boundaries != nullptr,
        "OrderScheme: boundaries must not be null",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        boundaries->is_available(),
        "OrderScheme: boundaries must be device-resident",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        keys.size() == static_cast<std::size_t>(boundaries->shape().second),
        "OrderScheme: number of keys must match number of boundary columns",
        std::invalid_argument
    );
    this->keys = std::move(keys);
    this->boundaries = std::move(boundaries);
    this->strict_boundaries = strict_boundaries;
}

PartitioningSpec PartitioningSpec::from_order(OrderScheme o) {
    return {.type = Type::ORDER, .hash = std::nullopt, .order = std::move(o)};
}

OrderScheme OrderScheme::replace_keys(std::vector<OrderKey> new_keys) const {
    return OrderScheme(std::move(new_keys), boundaries, strict_boundaries);
}

bool OrderScheme::boundaries_aligned_with(OrderScheme const& other) const {
    if (strict_boundaries != other.strict_boundaries)
        return false;
    if (boundaries->shape() != other.boundaries->shape())
        return false;
    if (boundaries->shape().first == 0)
        return true;
    auto const lhs = boundaries->table_view();
    auto const rhs = other.boundaries->table_view();
    auto const stream = boundaries->stream();
    // Ensure rhs data is visible on our stream before comparing
    other.boundaries->stream().synchronize();
    for (cudf::size_type i = 0; i < lhs.num_columns(); ++i) {
        auto eq = cudf::binary_operation(
            lhs.column(i),
            rhs.column(i),
            cudf::binary_operator::EQUAL,
            cudf::data_type{cudf::type_id::BOOL8},
            stream
        );
        auto result = cudf::reduce(
            eq->view(),
            *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
            cudf::data_type{cudf::type_id::BOOL8},
            stream
        );
        auto& scalar = static_cast<cudf::numeric_scalar<bool>&>(*result);
        if (!scalar.is_valid(stream) || !scalar.value(stream)) {
            return false;
        }
    }
    return true;
}

bool OrderScheme::operator==(OrderScheme const& other) const {
    return keys == other.keys && strict_boundaries == other.strict_boundaries
           && boundaries->shape() == other.boundaries->shape();
}

Message to_message(std::uint64_t sequence_number, std::unique_ptr<ChannelMetadata> m) {
    return Message{
        sequence_number,
        std::move(m),
        {},
        [](Message const& msg, MemoryReservation& /* reservation */) -> Message {
            auto copy = std::make_unique<ChannelMetadata>(msg.get<ChannelMetadata>());
            return Message{msg.sequence_number(), std::move(copy), {}, msg.copy_cb()};
        }
    };
}

}  // namespace rapidsmpf::streaming
