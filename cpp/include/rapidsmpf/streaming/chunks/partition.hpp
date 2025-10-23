/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/streaming/core/message.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Chunk of packed partitions identified by partition ID.
 *
 * Represents a single unit of work in a streaming pipeline where each partition
 * is associated with a `PartID` and contains packed (serialized) data.
 */
struct PartitionMapChunk {
    /**
     * @brief Sequence number used to preserve chunk ordering.
     */
    std::uint64_t sequence_number;

    /**
     * @brief Packed data for each partition, keyed by partition ID.
     */
    std::unordered_map<shuffler::PartID, PackedData> data;

    Message to_message() {
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
                    auto dst = br->allocate(
                        packed_data.data->size, packed_data.data->stream(), reservation
                    );
                    buffer_copy(*dst, *packed_data.data, packed_data.data->size);
                    ret.emplace(
                        pid,
                        PackedData{
                            std::make_unique<std::vector<std::uint8_t>>(
                                *packed_data.metadata
                            ),
                            std::move(dst)
                        }
                    );
                }
                return Message(
                    std::make_unique<PartitionMapChunk>(
                        self.sequence_number, std::move(ret)
                    ),
                    msg.callbacks()
                );
            }
        };
        return Message{
            std::make_unique<PartitionMapChunk>(std::move(*this)), std::move(cbs)
        };
    }
};

/**
 * @brief Chunk of packed partitions stored as a vector.
 *
 * Represents a single unit of work in a streaming pipeline where the partitions
 * are stored in a vector.
 */
struct PartitionVectorChunk {
    /**
     * @brief Sequence number used to preserve chunk ordering.
     */
    std::uint64_t sequence_number;

    /**
     * @brief Packed data for each partition stored in a vector.
     */
    std::vector<PackedData> data;
};


}  // namespace rapidsmpf::streaming
