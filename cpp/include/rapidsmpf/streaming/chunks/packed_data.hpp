/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/streaming/core/message.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Chunk of `PackedData` with sequence number.
 */
struct PackedDataChunk {
    /**
     * @brief Sequence number used to preserve chunk ordering.
     */
    std::uint64_t sequence_number;

    /**
     * @brief Packed data payload.
     */
    PackedData data;
};

Message to_message(PackedDataChunk&& chunk) {
    Message::Callbacks cbs{
        .buffer_size = [](Message const& msg, MemoryType mem_type) -> size_t {
            auto const& self = msg.get<PackedDataChunk>();
            if (self.data.data->mem_type() == mem_type) {
                return self.data.data->size;
            }
            return 0;
        },
        .copy = [](Message const& msg,
                   BufferResource* br,
                   MemoryReservation& reservation) -> Message {
            auto const& self = msg.get<PackedDataChunk>();
            return Message(
                std::make_unique<PackedDataChunk>(
                    self.sequence_number, self.data.copy(br, reservation)
                ),
                msg.callbacks()
            );
        }
    };
    return Message{std::make_unique<PackedDataChunk>(std::move(chunk)), std::move(cbs)};
}

}  // namespace rapidsmpf::streaming
