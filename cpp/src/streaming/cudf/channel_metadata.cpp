/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <memory>

#include <rapidsmpf/streaming/cudf/channel_metadata.hpp>

namespace rapidsmpf::streaming {

ContentDescription get_content_description(ChannelMetadata const& /* obj */) {
    // ChannelMetadata has negligible memory cost.
    return ContentDescription{};
}

Message to_message(std::uint64_t sequence_number, std::unique_ptr<ChannelMetadata> m) {
    auto cd = get_content_description(*m);
    return Message{
        sequence_number,
        std::move(m),
        cd,
        // Copy callback: ChannelMetadata is trivially copyable.
        [](Message const& msg, MemoryReservation& /* reservation */) -> Message {
            auto const& self = msg.get<ChannelMetadata>();
            auto copy = std::make_unique<ChannelMetadata>(self);
            auto cd = get_content_description(*copy);
            return Message{msg.sequence_number(), std::move(copy), cd, msg.copy_cb()};
        }
    };
}

}  // namespace rapidsmpf::streaming
