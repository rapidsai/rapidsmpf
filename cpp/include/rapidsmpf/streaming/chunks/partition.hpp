/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <unordered_map>
#include <vector>

#include <rapidsmpf/buffer/content_description.hpp>
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
     * @brief Packed data for each partition, keyed by partition ID.
     */
    std::unordered_map<shuffler::PartID, PackedData> data;
};

/**
 * @brief Chunk of packed partitions stored as a vector.
 *
 * Represents a single unit of work in a streaming pipeline where the partitions
 * are stored in a vector.
 */
struct PartitionVectorChunk {
    /**
     * @brief Packed data for each partition stored in a vector.
     */
    std::vector<PackedData> data;
};

/**
 * @brief Generate a content description for a `PartitionMapChunk`.
 *
 * @param obj The object's content to describe.
 * @return A new content description.
 */
ContentDescription get_content_description(PartitionMapChunk const& obj);

/**
 * @brief Generate a content description for a `PartitionVectorChunk`.
 *
 * @param obj The object's content to describe.
 * @return A new content description.
 */
ContentDescription get_content_description(PartitionVectorChunk const& obj);

/**
 * @brief Wrap a `PartitionMapChunk` into a `Message`.
 *
 * @param sequence_number Ordering identifier for the message.
 * @param chunk The chunk to wrap into a message.
 * @return A `Message` encapsulating the provided chunk as its payload.
 */
Message to_message(
    std::uint64_t sequence_number, std::unique_ptr<PartitionMapChunk> chunk
);

/**
 * @brief Wrap a `PartitionVectorChunk` into a `Message`.
 *
 * @param sequence_number Ordering identifier for the message.
 * @param chunk The chunk to wrap into a message.
 * @return A `Message` encapsulating the provided chunk as its payload.
 */
Message to_message(
    std::uint64_t sequence_number, std::unique_ptr<PartitionVectorChunk> chunk
);

}  // namespace rapidsmpf::streaming
