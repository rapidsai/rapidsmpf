/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/streaming/chunks/partition.hpp>

namespace rapidsmpf::streaming {

ContentDescription get_content_description(PartitionMapChunk const& obj) {
    ContentDescription ret{ContentDescription::Spillable::YES};
    for (auto const& [_, packed_data] : obj.data) {
        ret.content_size(packed_data.data->mem_type()) += packed_data.data->size;
    }
    return ret;
}

ContentDescription get_content_description(PartitionVectorChunk const& obj) {
    ContentDescription ret{ContentDescription::Spillable::YES};
    for (auto const& packed_data : obj.data) {
        ret.content_size(packed_data.data->mem_type()) += packed_data.data->size;
    }
    return ret;
}

Message to_message(
    std::uint64_t sequence_number, std::unique_ptr<PartitionMapChunk> chunk
) {
    auto cd = get_content_description(*chunk);
    return Message{
        sequence_number,
        std::move(chunk),
        cd,
        Message::Callbacks{
            .copy = [](Message const& msg, MemoryReservation& reservation) -> Message {
                auto const& self = msg.get<PartitionMapChunk>();
                std::unordered_map<shuffler::PartID, PackedData> pd;
                for (auto const& [pid, packed_data] : self.data) {
                    pd.emplace(pid, packed_data.copy(reservation));
                }
                auto chunk = std::make_unique<PartitionMapChunk>(std::move(pd));
                auto cd = get_content_description(*chunk);
                return Message{
                    msg.sequence_number(), std::move(chunk), cd, msg.callbacks()
                };
            },
            .move = [](Message&& msg, MemoryReservation& reservation) -> Message {
                auto callbacks = msg.callbacks();
                auto chunk =
                    std::make_unique<PartitionMapChunk>(msg.release<PartitionMapChunk>());
                for (auto& [_, packed_data] : chunk->data) {
                    packed_data.data =
                        reservation.br()->move(std::move(packed_data.data), reservation);
                }
                auto cd = get_content_description(*chunk);
                return Message{msg.sequence_number(), std::move(chunk), cd, callbacks};
            }
        }
    };
}

Message to_message(
    std::uint64_t sequence_number, std::unique_ptr<PartitionVectorChunk> chunk
) {
    auto cd = get_content_description(*chunk);
    return Message{
        sequence_number,
        std::move(chunk),
        cd,
        Message::Callbacks{
            .copy = [](Message const& msg, MemoryReservation& reservation) -> Message {
                auto const& self = msg.get<PartitionVectorChunk>();
                std::vector<PackedData> pd;
                for (auto const& packed_data : self.data) {
                    pd.emplace_back(packed_data.copy(reservation));
                }
                auto chunk = std::make_unique<PartitionVectorChunk>(std::move(pd));
                auto cd = get_content_description(*chunk);
                return Message{
                    msg.sequence_number(), std::move(chunk), cd, msg.callbacks()
                };
            },
            .move = [](Message&& msg, MemoryReservation& reservation) -> Message {
                auto callbacks = msg.callbacks();
                auto chunk = std::make_unique<PartitionVectorChunk>(
                    msg.release<PartitionVectorChunk>()
                );
                for (auto& packed_data : chunk->data) {
                    packed_data.data =
                        reservation.br()->move(std::move(packed_data.data), reservation);
                }
                auto cd = get_content_description(*chunk);
                return Message{msg.sequence_number(), std::move(chunk), cd, callbacks};
            }
        }
    };
}

}  // namespace rapidsmpf::streaming
