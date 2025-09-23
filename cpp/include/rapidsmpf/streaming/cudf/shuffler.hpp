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
 */
class ShufflerAsync {
  public:
    /**
     * @brief Constructs a new ShufflerAsync instance.
     *
     * @param ctx The streaming context to use.
     * @param stream The CUDA stream on which shuffling operations will be performed.
     * @param op_id Unique operation ID for this shuffle. Must not be reused until all
     * participants have completed the shuffle operation.
     * @param total_num_partitions Total number of partitions to shuffle data into.
     * @param partition_owner Function that maps a partition ID to its owning rank/node.
     * Defaults to round-robin distribution.
     */
    ShufflerAsync(
        std::shared_ptr<Context> ctx,
        rmm::cuda_stream_view stream,
        OpID op_id,
        shuffler::PartID total_num_partitions,
        shuffler::Shuffler::PartitionOwner partition_owner =
            shuffler::Shuffler::round_robin
    );

    // Prevent copying
    ShufflerAsync(ShufflerAsync const&) = delete;
    ShufflerAsync& operator=(ShufflerAsync const&) = delete;

    ~ShufflerAsync() = default;

    /**
     * @brief Gets the streaming context associated with this shuffler.
     *
     * @return A reference to the shared context object.
     */
    constexpr std::shared_ptr<Context> const& ctx() const {
        return ctx_;
    }

    /**
     * @brief Checks if the shuffle operation has completed.
     *
     * @return true if all partitions have been processed and the shuffle is complete,
     * false otherwise.
     */
    bool finished() const;

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
     * This coroutine will suspend until the specified partition is ready for extraction
     * (i.e., insert_finished has been called for this partition and all data has been
     * shuffled).
     *
     * @warning Users should be careful when using `extract_async` and `extract_any_async`
     * together, because a pid intended for `extract_async` may be extracted by
     * `extract_any_async`, hence there will be no guarantee that the chunks will be
     * returned. If that happens, `extract_async` will throw an std::out_of_range error.
     *
     * @param pid The partition ID to extract data for.
     * @return A vector of PackedData chunks for the partition.
     *
     * @throws std::out_of_range if the partition ID is not found or already extracted.
     *
     */
    coro::task<std::vector<PackedData>> extract_async(shuffler::PartID pid);

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
     * @return ExtractResult containing the partition ID and data chunks, or a nullopt
     * if no more partitions are available.
     *
     * @warning Users should be careful when using `extract_async` and `extract_any_async`
     * together, because a pid intended for `extract_async` may be extracted by
     * `extract_any_async`.
     */
    coro::task<std::optional<ExtractResult>> extract_any_async();

  private:
    /// @brief Checks if all local partitions have been extracted (not thread safe).
    [[nodiscard]] bool all_extracted_unsafe() const;

    coro::mutex mtx_{};
    coro::condition_variable cv_{};
    std::shared_ptr<Context> ctx_;
    shuffler::Shuffler shuffler_;
    std::unordered_set<shuffler::PartID>
        ready_pids_;  ///< set to collect all the partitions that are ready for
                      ///< extraction. It will be untimately be emptied once all local
                      ///< partitions have been extracted.
    std::unordered_set<shuffler::PartID>
        extracted_pids_;  ///< set to collect all the partitions that have been extracted.
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
 * @param stream The CUDA stream on which to perform the shuffling. If chunks from the
 * input channel aren't created on `stream`, the streams are all synchronized.
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
    rmm::cuda_stream_view stream,
    std::shared_ptr<Channel> ch_in,
    std::shared_ptr<Channel> ch_out,
    OpID op_id,
    shuffler::PartID total_num_partitions,
    shuffler::Shuffler::PartitionOwner partition_owner = shuffler::Shuffler::round_robin
);

}  // namespace node

}  // namespace rapidsmpf::streaming
