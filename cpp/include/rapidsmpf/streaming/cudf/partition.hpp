/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>

#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>
#include <rapidsmpf/streaming/cudf/utils.hpp>

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
 * @param ch_in Input channel providing tables to partition.
 * @param ch_out Output channel to which packed partitioned table are sent.
 * @param columns_to_hash Indices of input columns to hash.
 * @param num_partitions The number of partitions to use.
 * @param hash_function Hash function to use for partitioning.
 * @param seed Seed value for the hash function.
 *
 * @return Streaming node representing the asynchronous partitioning and packing
 * operation.
 *
 * @throw std::out_of_range if any index in `columns_to_hash` is invalid.
 *
 * @see rapidsmpf::partition_and_split
 */
Node partition_and_pack(
    std::shared_ptr<Context> ctx,
    SharedChannel<TableChunk> ch_in,
    SharedChannel<PartitionMapChunk> ch_out,
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
 * @param ch_in Input channel providing packed partition chunks.
 * @param ch_out Output channel to which unpacked and concatenated tables table are sent.
 *
 * @return Streaming node representing the asynchronous unpacking and concatenation
 * operation.
 *
 * @see rapidsmpf::unpack_and_concat
 */
Node unpack_and_concat(
    std::shared_ptr<Context> ctx,
    SharedChannel<PartitionMapChunk> ch_in,
    SharedChannel<TableChunk> ch_out
);


/**
 * @brief Asynchronously unpacks and concatenates vectors of packed partitions.
 *
 * This is a streaming version of `rapidsmpf::unpack_and_concat` that operates on
 * vectors of packed partition chunks using channels.
 *
 * It receives vectors of packed partitions from the input channel, deserializes and
 * concatenates them, and sends the resulting tables to the output channel. Empty
 * partitions are ignored.
 *
 * @param ctx The context to use.
 * @param ch_in Input channel providing vectors of packed partition chunks.
 * @param ch_out Output channel to which unpacked and concatenated tables table are sent.
 *
 * @return Streaming node representing the asynchronous unpacking and concatenation
 * operation.
 *
 * @see rapidsmpf::unpack_and_concat
 */
Node unpack_and_concat(
    std::shared_ptr<Context> ctx,
    SharedChannel<PartitionVectorChunk> ch_in,
    SharedChannel<TableChunk> ch_out
);

}  // namespace node
}  // namespace rapidsmpf::streaming
