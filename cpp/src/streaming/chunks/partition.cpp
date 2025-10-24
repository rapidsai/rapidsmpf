/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/streaming/chunks/partition.hpp>

namespace rapidsmpf::streaming {

Message to_message(PartitionMapChunk&& chunk) {
    Message::Callbacks cbs{
        .buffer_size = [](Message const& msg, MemoryType mem_type) -> size_t {
            auto const& self = msg.get<PartitionMapChunk>();
            std::size_t ret = 0;
            for (auto const& [_, packed_data] : self.data) {
                if (mem_type == packed_data.data->mem_type()) {
                    ret += packed_data.data->size;
                }
            }
            return ret;
        },
        .copy = [](Message const& msg,
                   BufferResource* br,
                   MemoryReservation& reservation) -> Message {
            auto const& self = msg.get<PartitionMapChunk>();
            std::unordered_map<shuffler::PartID, PackedData> ret;
            for (auto const& [pid, packed_data] : self.data) {
                ret.emplace(pid, packed_data.copy(br, reservation));
            }
            return Message(
                std::make_unique<PartitionMapChunk>(self.sequence_number, std::move(ret)),
                msg.callbacks()
            );
        }
    };
    return Message{std::make_unique<PartitionMapChunk>(std::move(chunk)), std::move(cbs)};
}

}  // namespace rapidsmpf::streaming
