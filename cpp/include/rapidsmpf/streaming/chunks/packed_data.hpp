/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rapidsmpf/buffer/content_description.hpp>
#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/streaming/core/message.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Chunk of `PackedData`.
 */
struct PackedDataChunk {
    /**
     * @brief Packed data payload.
     */
    PackedData data;
};

/**
 * @brief Generate a content description for a `PackedDataChunk`.
 *
 * @param obj The object's content to describe.
 * @return A new content description.
 */
inline ContentDescription get_content_description(PackedDataChunk const& obj) {
    return ContentDescription{
        {{obj.data.data->mem_type(), obj.data.data->size}},
        ContentDescription::Spillable::YES
    };
}

/**
 * @brief Wrap a `PackedDataChunk` into a `Message`.
 *
 * @param sequence_number Ordering identifier for the message.
 * @param chunk The chunk to wrap into a message.
 * @return A `Message` encapsulating the provided chunk as its payload.
 */
Message to_message(
    std::uint64_t sequence_number, std::unique_ptr<PackedDataChunk> chunk
) {
    auto cd = get_content_description(*chunk);
    return Message{
        sequence_number,
        std::move(chunk),
        cd,
        [](Message const& msg, MemoryReservation& reservation) -> Message {
            auto const& self = msg.get<PackedDataChunk>();
            auto chunk = std::make_unique<PackedDataChunk>(self.data.copy(reservation));
            auto cd = get_content_description(*chunk);
            return Message{msg.sequence_number(), std::move(chunk), cd, msg.copy_cb()};
        }
    };
}

}  // namespace rapidsmpf::streaming
