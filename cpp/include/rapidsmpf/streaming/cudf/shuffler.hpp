/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <unordered_set>

#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/partition.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief An asynchronous shuffler that allows concurrent insertion and extraction of
 * data.
 *
 * ShufflerAsync provides an asynchronous interface to the shuffler, allowing data to be
 * inserted while previously shuffled partitions are extracted concurrently. This is
 * useful for streaming scenarios where data can be processed as soon as individual
 * partitions are ready, rather than waiting for the entire shuffle to complete.
 *
 * @note Unlike the synchronous shuffler, this class does not provide a `finished()`
 * method. Instead, completion is implied: extraction coroutines (`extract_async`,
 * `extract_any_async`) will return `std::nullopt` when no more partitions are
 * available.
 */
class ShufflerAsync {
  public:
    /**
     * @brief Constructs a new ShufflerAsync instance.
     *
     * @param ctx The streaming context to use.
     * @param op_id Unique operation ID for this shuffle. Must not be reused until all
     * participants have completed the shuffle operation.
     * @param total_num_partitions Total number of partitions to shuffle data into.
     * @param partition_owner Function that maps a partition ID to its owning rank/node.
     * Defaults to round-robin distribution.
     *
     * @note The caller promises that inserted buffers are stream-ordered with respect
     * to their own stream, and extracted buffers are likewise guaranteed to be stream-
     * ordered with respect to their own stream.
     */
    ShufflerAsync(
        std::shared_ptr<Context> ctx,
        OpID op_id,
        shuffler::PartID total_num_partitions,
        shuffler::Shuffler::PartitionOwner partition_owner =
            shuffler::Shuffler::round_robin
    );

    // Prevent copying
    ShufflerAsync(ShufflerAsync const&) = delete;
    ShufflerAsync& operator=(ShufflerAsync const&) = delete;

    ~ShufflerAsync() noexcept;

    /**
     * @brief Gets the streaming context associated with this shuffler.
     *
     * @return A reference to the shared context object.
     */
    constexpr std::shared_ptr<Context> const& ctx() const {
        return ctx_;
    }

    /**
     * @brief Gets the total number of partitions for this shuffle operation.
     *
     * @return The total number of partitions that data will be shuffled into.
     */
    constexpr shuffler::PartID total_num_partitions() const {
        return shuffler_.total_num_partitions;
    }

    /**
     * @brief Gets the partition owner function used by this shuffler.
     *
     * @return A const reference to the function that maps partition IDs to owning ranks.
     */
    constexpr shuffler::Shuffler::PartitionOwner const& partition_owner() const {
        return shuffler_.partition_owner;
    }

    /// @copydoc rapidsmpf::shuffler::Shuffler::local_partitions
    [[nodiscard]] std::span<shuffler::PartID const> local_partitions() const;

    /// @copydoc rapidsmpf::shuffler::Shuffler::insert
    void insert(std::unordered_map<shuffler::PartID, PackedData>&& chunks);

    /// @copydoc rapidsmpf::shuffler::Shuffler::insert_finished(std::vector<PartID>&&)
    void insert_finished(std::vector<shuffler::PartID>&& pids);

    /**
     * @brief Asynchronously extracts all data for a specific partition.
     *
     * This coroutine suspends until the specified partition is ready for extraction
     * (i.e., `insert_finished` has been called for this partition and all data has been
     * shuffled).
     *
     * @warning Be careful when mixing `extract_async` and `extract_any_async`.
     * A partition intended for `extract_async` may already have been consumed by
     * `extract_any_async`, in which case this function returns `std::nullopt`.
     *
     * @param pid The partition ID to extract data for.
     * @return
     *   - `std::nullopt` if the partition ID is not ready or has already been extracted.
     *   - Otherwise, a vector of `PackedData` chunks belonging to the partition.
     *
     * @throws std::out_of_range If the partition ID isn't owned by this rank, see
     * `partition_owner()`.
     */
    coro::task<std::optional<std::vector<PackedData>>> extract_async(
        shuffler::PartID pid
    );

    /**
     * @brief Result type for extract_any_async operations.
     *
     * Contains the partition ID and associated data chunks from an extract operation.
     */
    using ExtractResult = std::pair<shuffler::PartID, std::vector<PackedData>>;

    /**
     * @brief Asynchronously extracts data for any ready partition.
     *
     * This coroutine will suspend until at least one partition is ready for extraction,
     * then extract and return the data for one such partition. If no partitions become
     * ready and the shuffle is finished, returns a nullopt.
     *
     * @return `ExtractResult` containing the partition ID and data chunks, or a nullopt
     * if all partitions has been extracted.
     *
     * @warning Be careful when mixing `extract_async` and `extract_any_async`.
     * A partition intended for `extract_async` may already have been consumed by
     * `extract_any_async`, in which case `extract_async` will later return
     * `std::nullopt`.
     */
    coro::task<std::optional<ExtractResult>> extract_any_async();

  private:
    coro::mutex mtx_{};
    coro::condition_variable cv_{};
    std::shared_ptr<Context> ctx_;
    shuffler::Shuffler shuffler_;

    /**
     * @brief Tracks partition states for extraction.
     *
     * A received partition's ID is always in exactly one of the two sets:
     *   - `ready_pids_`: partitions ready for extraction but not yet extracted.
     *   - `extracted_pids_`: partitions that have already been extracted.
     */
    std::unordered_set<shuffler::PartID> ready_pids_;
    std::unordered_set<shuffler::PartID> extracted_pids_;
};

namespace node {
/**
 * @brief Launches a shuffler node for a single shuffle operation.
 *
 * This is a streaming version of `rapidsmpf::shuffler::Shuffler` that operates on
 * packed partition chunks using channels.
 *
 * It consumes partitioned input data from the input channel and produces output
 * chunks grouped by `partition_owner`.
 *
 * @param ctx The context to use.
 * @param ch_in Input channel providing PartitionMapChunk to be shuffled.
 * @param ch_out Output channel where the resulting PartitionVectorChunks are sent.
 * @param op_id Unique operation ID for this shuffle. Must not be reused until all
 * nodes have called `Shuffler::shutdown()`.
 * @param total_num_partitions Total number of partitions to shuffle the data into.
 * @param partition_owner Function that maps a partition ID to its owning rank/node.
 *
 * @return A streaming node that completes when the shuffling has finished and the
 * output channel is drained.
 */
Node shuffler(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::shared_ptr<Channel> ch_out,
    OpID op_id,
    shuffler::PartID total_num_partitions,
    shuffler::Shuffler::PartitionOwner partition_owner = shuffler::Shuffler::round_robin
);

}  // namespace node

}  // namespace rapidsmpf::streaming
