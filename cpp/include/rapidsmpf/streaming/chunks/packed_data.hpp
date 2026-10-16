/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rapidsmpf/memory/content_description.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/streaming/core/message.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Generate a content description for `PackedData`.
 *
 * @param obj The object's content to describe.
 * @return A new content description.
 */
inline ContentDescription get_content_description(PackedData const& obj) {
    return ContentDescription{
        {{obj.data->mem_type(), obj.data->size}}, ContentDescription::Spillable::YES
    };
}

/**
 * @brief Wrap `PackedData` into a `Message`.
 *
 * @param sequence_number Ordering identifier for the message.
 * @param chunk The chunk to wrap into a message.
 * @return A `Message` encapsulating the provided chunk as its payload.
 */
inline Message to_message(
    std::uint64_t sequence_number, std::unique_ptr<PackedData> chunk
) {
    auto cd = get_content_description(*chunk);
    return Message{
        sequence_number,
        std::move(chunk),
        cd,
        Message::Callbacks{
            .copy = [](Message const& msg, MemoryReservation& reservation) -> Message {
                auto const& self = msg.get<PackedData>();
                auto chunk = std::make_unique<PackedData>(self.copy(reservation));
                auto cd = get_content_description(*chunk);
                return Message{
                    msg.sequence_number(), std::move(chunk), cd, msg.callbacks()
                };
            },
            .move = [](Message&& msg, MemoryReservation& reservation) -> Message {
                auto callbacks = msg.callbacks();
                auto chunk = std::make_unique<PackedData>(msg.release<PackedData>());
                chunk->data = reservation.br()->move(std::move(chunk->data), reservation);
                auto cd = get_content_description(*chunk);
                return Message{msg.sequence_number(), std::move(chunk), cd, callbacks};
            }
        }
    };
}

}  // namespace rapidsmpf::streaming
