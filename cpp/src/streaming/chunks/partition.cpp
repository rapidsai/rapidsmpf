/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/streaming/chunks/partition.hpp>

namespace rapidsmpf::streaming {

ContentDescription get_content_description(PartitionMapChunk const& obj) {
    ContentDescription ret{/* spillable = */ true};
    for (auto const& [_, packed_data] : obj.data) {
        ret.content_size(packed_data.data->mem_type()) += packed_data.data->size;
    }
    return ret;
}

ContentDescription get_content_description(PartitionVectorChunk const& obj) {
    ContentDescription ret{/* spillable = */ true};
    for (auto const& packed_data : obj.data) {
        ret.content_size(packed_data.data->mem_type()) += packed_data.data->size;
    }
    return ret;
}

Message to_message(
    std::uint64_t sequence_number, std::unique_ptr<PartitionMapChunk> chunk
) {
    Message::Callbacks cbs{
        .content_size = [](Message const& msg,
                           MemoryType mem_type) -> std::pair<size_t, bool> {
            auto const& self = msg.get<PartitionMapChunk>();
            std::size_t ret = 0;
            for (auto const& [_, packed_data] : self.data) {
                if (mem_type == packed_data.data->mem_type()) {
                    ret += packed_data.data->size;
                }
            }
            return {ret, true};
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
                msg.sequence_number(),
                std::make_unique<PartitionMapChunk>(std::move(ret)),
                msg.callbacks()
            );
        }
    };
    return Message{sequence_number, std::move(chunk), std::move(cbs)};
}

Message to_message(
    std::uint64_t sequence_number, std::unique_ptr<PartitionVectorChunk> chunk
) {
    Message::Callbacks cbs{
        .content_size = [](Message const& msg,
                           MemoryType mem_type) -> std::pair<size_t, bool> {
            auto const& self = msg.get<PartitionVectorChunk>();
            std::size_t ret = 0;
            for (auto const& packed_data : self.data) {
                if (mem_type == packed_data.data->mem_type()) {
                    ret += packed_data.data->size;
                }
            }
            return {ret, true};
        },
        .copy = [](Message const& msg,
                   BufferResource* br,
                   MemoryReservation& reservation) -> Message {
            auto const& self = msg.get<PartitionVectorChunk>();
            std::vector<PackedData> ret;
            for (auto const& packed_data : self.data) {
                ret.emplace_back(packed_data.copy(br, reservation));
            }
            return Message(
                msg.sequence_number(),
                std::make_unique<PartitionVectorChunk>(std::move(ret)),
                msg.callbacks()
            );
        }
    };
    return Message{sequence_number, std::move(chunk), std::move(cbs)};
}

}  // namespace rapidsmpf::streaming
