/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>

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
