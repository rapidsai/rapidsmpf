/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

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
 * @brief Wrap a `PackedDataChunk` into a `Message`.
 *
 * @param sequence_number Ordering identifier for the message.
 * @param chunk The chunk to wrap into a message.
 * @return A `Message` encapsulating the provided chunk as its payload.
 */
Message to_message(
    std::uint64_t sequence_number, std::unique_ptr<PackedDataChunk> chunk
) {
    Message::Callbacks cbs{
        .primary_data_size = [](Message const& msg,
                                MemoryType mem_type) -> std::pair<size_t, bool> {
            auto const& self = msg.get<PackedDataChunk>();
            if (self.data.data->mem_type() == mem_type) {
                return {self.data.data->size, true};
            }
            return {0, true};
        },
        .copy = [](Message const& msg,
                   BufferResource* br,
                   MemoryReservation& reservation) -> Message {
            auto const& self = msg.get<PackedDataChunk>();
            return Message(
                msg.sequence_number(),
                std::make_unique<PackedDataChunk>(self.data.copy(br, reservation)),
                msg.callbacks()
            );
        }
    };
    return Message{sequence_number, std::move(chunk), std::move(cbs)};
}

}  // namespace rapidsmpf::streaming
