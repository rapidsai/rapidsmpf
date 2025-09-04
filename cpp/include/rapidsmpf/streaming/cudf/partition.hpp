/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <unordered_map>
#include <vector>

#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>

#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Chunk of packed partitions identified by partition ID.
 *
 * Represents a single unit of work in a streaming pipeline where each partition
 * is associated with a `PartID` and contains packed (serialized) data.
 *
 * The `sequence_number` is used to preserve ordering across chunks.
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

    /**
     * @brief The CUDA stream on which this chunk was created.
     */
    rmm::cuda_stream_view stream;
};

/**
 * @brief Chunk of packed partitions stored as a vector.
 *
 * Represents a single unit of work in a streaming pipeline where the partitions
 * are stored in a vector.
 *
 * The `sequence_number` is used to preserve ordering across chunks.
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

    /**
     * @brief The CUDA stream on which this chunk was created.
     */
    rmm::cuda_stream_view stream;
};

namespace node {

/**
 * @brief Asynchronously partitions input tables into multiple packed (serialized) tables.
 *
 * This is a streaming version of `rapidsmpf::partition_and_split` that operates on table
 * chunks using channels.
 *
 * It receives tables from an input channel, partitions each row into one of
 * `num_partitions` based on a hash of the selected columns, packs the resulting
 * partitions, and sends them to an output channel.
 *
 * @param ctx The context to use.
 * @param ch_in Input channel providing `TableChunk`s to partition.
 * @param ch_out Output channel to which `PartitionMapChunk`s are sent.
 * @param columns_to_hash Indices of input columns to hash.
 * @param num_partitions The number of partitions to use.
 * @param hash_function Hash function to use for partitioning.
 * @param seed Seed value for the hash function.
 *
 * @return Streaming node representing the asynchronous partitioning and packing
 * operation.
 *
 * @throws std::out_of_range if any index in `columns_to_hash` is invalid.
 *
 * @see rapidsmpf::partition_and_split
 */
Node partition_and_pack(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::shared_ptr<Channel> ch_out,
    std::vector<cudf::size_type> columns_to_hash,
    int num_partitions,
    cudf::hash_id hash_function,
    uint32_t seed
);


/**
 * @brief Asynchronously unpacks and concatenates packed partitions.
 *
 * This is a streaming version of `rapidsmpf::unpack_and_concat` that operates on
 * packed partition chunks using channels.
 *
 * It receives packed partitions from the input channel, deserializes and concatenates
 * them, and sends the resulting tables to the output channel. Empty partitions are
 * ignored.
 *
 * @param ctx The context to use.
 * @param ch_in Input channel providing packed partitions as PartitionMapChunk or
 * PartitionVectorChunk.
 * @param ch_out Output channel to which unpacked and concatenated tables table are sent.
 *
 * @return Streaming node representing the asynchronous unpacking and concatenation
 * operation.
 *
 * @see rapidsmpf::unpack_and_concat
 */
Node unpack_and_concat(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::shared_ptr<Channel> ch_out
);

}  // namespace node
}  // namespace rapidsmpf::streaming
