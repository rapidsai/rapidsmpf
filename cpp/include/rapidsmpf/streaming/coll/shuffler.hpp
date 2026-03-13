/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <coro/event.hpp>

#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/cudf/partition.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief An asynchronous shuffler that wraps the synchronous shuffler with a coroutine
 * interface.
 *
 * ShufflerAsync provides an asynchronous interface to the shuffler, allowing data to be
 * inserted and then extracted after the shuffle completes. All local partitions complete
 * simultaneously, so extraction is non-blocking after awaiting `insert_finished()`.
 *
 * @warning The coroutine returned by `insert_finished()` _must_ be awaited before the
 * object is destroyed, otherwise the shuffle with terminate in destruction and/or
 * deadlocks will occur.
 *
 * Example usage:
 * @code{.cpp}
 * auto shuffle = ShufflerAsync(...);
 * while (...) {
 *   shuffle.insert(...);
 * }
 * co_await shuffle.insert_finished();
 * for (auto pid : shuffle.local_partitions()) {
 *   auto chunks = shuffle.extract(pid);
 *   // process chunks...
 * }
 * @endcode{}
 */
class ShufflerAsync {
  public:
    /**
     * @brief Constructs a new ShufflerAsync instance.
     *
     * @param ctx The streaming context to use.
     * @param comm Communicator for the collective operation.
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
        std::shared_ptr<Communicator> comm,
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
    [[nodiscard]] constexpr std::shared_ptr<Context> const& ctx() const {
        return ctx_;
    }

    /**
     * @brief Gets the communicator associated with this shuffler.
     *
     * @return Shared pointer to communicator.
     */
    [[nodiscard]] std::shared_ptr<Communicator> const& comm() const noexcept {
        return shuffler_.comm();
    }

    /**
     * @brief Gets the total number of partitions for this shuffle operation.
     *
     * @return The total number of partitions that data will be shuffled into.
     */
    [[nodiscard]] constexpr shuffler::PartID total_num_partitions() const {
        return shuffler_.total_num_partitions;
    }

    /**
     * @brief Gets the partition owner function used by this shuffler.
     *
     * @return A const reference to the function that maps partition IDs to owning ranks.
     */
    [[nodiscard]] constexpr shuffler::Shuffler::PartitionOwner const&
    partition_owner() const {
        return shuffler_.partition_owner;
    }

    /// @copydoc rapidsmpf::shuffler::Shuffler::local_partitions
    [[nodiscard]] std::span<shuffler::PartID const> local_partitions() const;

    /// @copydoc rapidsmpf::shuffler::Shuffler::insert
    void insert(std::unordered_map<shuffler::PartID, PackedData>&& chunks);

    /**
     * @copydoc rapidsmpf::shuffler::Shuffler::insert_finished()
     *
     * @note This coroutine function must be awaited to ensure the shuffler has fully
     * completed its asynchronous operations.
     *
     * @return A coroutine that inserts the finish marker and suspends until the shuffle
     * has completed. Once complete,
     */
    [[nodiscard]] Actor insert_finished();

    /**
     * @brief Extract all chunks belonging to the specified partition.
     *
     * @param pid The ID of the partition to extract.
     * @throws std::logic_error If the partition has already been extracted or is
     * otherwise not available.
     * @return A vector of PackedData chunks associated with the partition.
     */
    [[nodiscard]] std::vector<PackedData> extract(shuffler::PartID pid);

  private:
    std::shared_ptr<Context> ctx_;
    coro::event
        event_{};  ///< Event tracking whether all data has arrived and can be extracted.
    shuffler::Shuffler shuffler_;
};

namespace actor {
/**
 * @brief Launches a shuffler actor for a single shuffle operation.
 *
 * This is a streaming version of `rapidsmpf::shuffler::Shuffler` that operates on
 * packed partition chunks using channels.
 *
 * It consumes partitioned input data from the input channel and produces output
 * chunks grouped by `partition_owner`.
 *
 * @param ctx The context to use.
 * @param comm Communicator for the collective operation.
 * @param ch_in Input channel providing PartitionMapChunk to be shuffled.
 * @param ch_out Output channel where the resulting PartitionVectorChunks are sent.
 * @param op_id Unique operation ID for this shuffle. Must not be reused until all
 * actors have called `Shuffler::shutdown()`.
 * @param total_num_partitions Total number of partitions to shuffle the data into.
 * @param partition_owner Function that maps a partition ID to its owning rank/node.
 *
 * @return A streaming actor that completes when the shuffling has finished and the
 * output channel is drained.
 */
[[nodiscard]] Actor shuffler(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Communicator> comm,
    std::shared_ptr<Channel> ch_in,
    std::shared_ptr<Channel> ch_out,
    OpID op_id,
    shuffler::PartID total_num_partitions,
    shuffler::Shuffler::PartitionOwner partition_owner = shuffler::Shuffler::round_robin
);

}  // namespace actor

}  // namespace rapidsmpf::streaming
