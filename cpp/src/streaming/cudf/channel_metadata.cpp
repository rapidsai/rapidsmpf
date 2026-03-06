/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>

#include <rapidsmpf/streaming/cudf/channel_metadata.hpp>

namespace rapidsmpf::streaming {

bool OrderScheme::operator==(OrderScheme const& other) const {
    // Compare basic fields
    if (column_indices != other.column_indices || orders != other.orders
        || null_orders != other.null_orders)
    {
        return false;
    }

    // Compare boundaries
    bool this_has_boundaries = boundaries != nullptr;
    bool other_has_boundaries = other.boundaries != nullptr;

    if (this_has_boundaries != other_has_boundaries) {
        return false;
    }

    if (!this_has_boundaries) {
        // Both are null
        return true;
    }

    // Both have boundaries - compare dimensions via table_view
    auto this_view = boundaries->table_view();
    auto other_view = other.boundaries->table_view();
    if (this_view.num_rows() != other_view.num_rows()
        || this_view.num_columns() != other_view.num_columns())
    {
        return false;
    }

    // Note: Full content comparison would require device-side comparison.
    // For now, we consider tables with same dimensions as potentially equal.
    // A more complete implementation would use cudf utilities for comparison.
    return true;
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
