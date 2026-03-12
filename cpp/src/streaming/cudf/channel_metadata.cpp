/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>

#include <rapidsmpf/streaming/cudf/channel_metadata.hpp>

namespace rapidsmpf::streaming {

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
