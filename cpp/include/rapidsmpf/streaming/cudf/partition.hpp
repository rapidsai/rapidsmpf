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
#include <rapidsmpf/streaming/core/base_chunk.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Chunk of packed partitions identified by partition ID.
 *
 * Represents a single unit of work in a streaming pipeline where each entry
 * maps a `shuffler::PartID` to its packed (serialized) data.
 *
 * The sequence number from BaseChunk is used to preserve cross-chunk ordering.
 */
class PartitionMapChunk : public BaseChunk {
  public:
    /**
     * @brief Construct a chunk with a partition map.
     *
     * @param sequence_number Ordering identifier for the chunk.
     * @param stream CUDA stream associated with buffers referenced by the packed data.
     * @param data The partition → packed-data map to take ownership of (moved).
     */
    PartitionMapChunk(
        std::uint64_t sequence_number,
        std::unordered_map<shuffler::PartID, PackedData>&& data,
        rmm::cuda_stream_view stream
    )
        : BaseChunk(sequence_number, stream), data_(std::move(data)) {}

    ~PartitionMapChunk() override = default;

    /**
     * @brief move constructor
     */
    PartitionMapChunk(PartitionMapChunk&&) noexcept = default;

    /**
     * @brief move assignment operator
     * @return this chunk.
     */
    PartitionMapChunk& operator=(PartitionMapChunk&&) noexcept = default;
    PartitionMapChunk(PartitionMapChunk const&) = delete;
    PartitionMapChunk& operator=(PartitionMapChunk const&) = delete;

    /**
     * @brief Access the packed data for each partition.
     *
     * @return Mutable reference to the partition → packed-data map.
     */
    [[nodiscard]] constexpr auto& data() noexcept {
        return data_;
    }

  private:
    std::unordered_map<shuffler::PartID, PackedData> data_;
};

/**
 * @brief Chunk of packed partitions stored as a vector.
 *
 * Represents a single unit of work in a streaming pipeline where the partitions
 * are stored in a vector.
 */
class PartitionVectorChunk : public BaseChunk {
  public:
    /**
     * @brief Chunk of packed partitions stored as a vector.
     *
     * @param sequence_number Ordering identifier for the chunk.
     * @param stream CUDA stream associated with buffers referenced by the packed data.
     * @param data The packed-data vector to take ownership of (moved).
     */
    PartitionVectorChunk(
        std::uint64_t sequence_number,
        std::vector<PackedData>&& data,
        rmm::cuda_stream_view stream
    )
        : BaseChunk(sequence_number, stream), data_(std::move(data)) {}

    ~PartitionVectorChunk() override = default;

    /**
     * @brief move constructor
     */
    PartitionVectorChunk(PartitionVectorChunk&&) noexcept = default;

    /**
     * @brief move assignment operator
     * @return this chunk.
     */
    PartitionVectorChunk& operator=(PartitionVectorChunk&&) noexcept = default;
    PartitionVectorChunk(PartitionVectorChunk const&) = delete;
    PartitionVectorChunk& operator=(PartitionVectorChunk const&) = delete;

    /**
     * @brief Access the packed data.
     *
     * @return Packed data for each partition stored in a vector.
     */
    [[nodiscard]] auto& data() noexcept {
        return data_;
    }

  private:
    std::vector<PackedData> data_;
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
