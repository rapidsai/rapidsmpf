/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_set>
#include <vector>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>

/**
 * @namespace rapidsmpf::shuffler
 * @brief Shuffler interfaces.
 *
 * A shuffle service for cuDF tables. Use `Shuffler` to perform a single shuffle.
 */
namespace rapidsmpf::shuffler {

/**
 * @namespace rapidsmpf::shuffler::detail
 * @brief Shuffler private interfaces.
 *
 * This namespace contains private interfaces for internal workings of the shuffler.
 * These interfaces may change without notice and should not be relied upon directly.
 */
namespace detail {

/**
 * @brief Helper to tally the finish status of a shuffle.
 *
 * The `FinishCounter` class tracks the completion of shuffle operations across different
 * ranks. Each rank reports its total chunk count once, and individual chunks are tallied
 * globally. All local partitions finish simultaneously when every rank has reported and
 * all chunks have been received.
 */
class FinishCounter {
  public:
    /**
     * @brief Callback function type called when a partition is finished.
     *
     * The callback receives the partition ID of the finished partition.
     *
     * @warning A callback must be fast and non-blocking and should not call any of the
     * `wait*` methods. And be very careful if acquiring locks. Ideally it should be used
     * to signal a separate thread to do the actual processing.
     */
    using FinishedCallback = std::function<void(PartID)>;

    /**
     * @brief Construct a finish counter.
     *
     * @param nranks The total number of ranks participating in the shuffle.
     * @param local_partitions The partition IDs local to the current rank.
     * @param finished_callback The callback to notify when a partition is finished
     * (optional).
     */
    FinishCounter(
        Rank nranks,
        std::span<PartID const> local_partitions,
        FinishedCallback&& finished_callback = nullptr
    );

    ~FinishCounter() = default;

    /**
     * @brief Move the goalpost for a specific source rank.
     *
     * This function sets the number of chunks that need to be received from a specific
     * rank. It should only be called once per rank.
     *
     * @param src_rank The source rank reporting its chunk count.
     * @param nchunks The number of chunks required. (Requires nchunks > 0)
     *
     * @throws std::logic_error If the goalpost is moved more than once for the same rank,
     * or if nchunks is 0.
     */
    void move_goalpost(Rank src_rank, ChunkID nchunks);

    /**
     * @brief Add a finished chunk to the global counter.
     *
     * This function increments the global finished chunk counter.
     * When all ranks have reported and the number of finished chunks matches the total
     * goal, all partitions are marked as finished.
     *
     * @throws std::logic_error If the total finished chunks exceed the goal.
     */
    void add_finished_chunk();

    /**
     * @brief Returns whether all partitions are finished (non-blocking).
     *
     * @return True if all partitions are finished, otherwise False.
     */
    [[nodiscard]] bool all_finished() const;

    /**
     * @brief Returns the partition ID of a finished partition that hasn't been waited on
     * (blocking). Optionally a timeout (in ms) can be provided.
     *
     * This function blocks until all partitions are finished and ready to be processed.
     * If the timeout is set and a partition is not available by the time, a
     * std::runtime_error will be thrown.
     *
     * @param timeout Optional timeout (ms) to wait.
     *
     * @return The partition ID of a finished partition.
     *
     * @throws std::out_of_range If all partitions have already been waited on.
     * @throws std::runtime_error If timeout was set and no partitions have been finished
     * by the expiration.
     */
    PartID wait_any(std::optional<std::chrono::milliseconds> timeout = {});

    /**
     * @brief Wait for a specific partition to be finished (blocking). Optionally a
     * timeout (in ms) can be provided.
     *
     * This function blocks until all partitions are finished and the desired partition
     * is ready to be processed. If the timeout is set and the requested partition is not
     * available by the time, a std::runtime_error will be thrown.
     *
     * @param pid The desired partition ID.
     * @param timeout Optional timeout (ms) to wait.
     *
     * @throws std::out_of_range If the desired partition is unavailable.
     * @throws std::runtime_error If timeout was set and requested partition has been
     * finished by the expiration.
     */
    void wait_on(PartID pid, std::optional<std::chrono::milliseconds> timeout = {});

    /**
     * @brief Returns a description of this instance.
     * @return The description.
     */
    [[nodiscard]] std::string str() const;

  private:
    Rank const nranks_;
    PartID
        n_unfinished_partitions_;  ///< aux counter to track the number of unfinished
                                   ///< partitions (without using the goalposts.empty())

    Rank n_ranks_with_goalpost_{0};  ///< how many ranks have called move_goalpost
    ChunkID total_chunk_goal_{0};  ///< sum of all rank chunk goals
    ChunkID total_finished_chunks_{0};  ///< global finished chunk counter
    std::vector<bool> rank_reported_;  ///< indexed by rank, prevents double-reporting
    std::span<PartID const> local_partitions_;  ///< for firing callbacks
    std::unordered_set<PartID>
        pending_pids_;  ///< partitions not yet consumed by wait_any/wait_on
    bool all_done_{false};  ///< prevents double-firing of callbacks

    mutable std::mutex mutex_;  // TODO: use a shared_mutex lock?
    mutable std::condition_variable wait_cv_;

    FinishedCallback finished_callback_ =
        nullptr;  ///< callback to notify when a partition is finished
};

}  // namespace detail

/**
 * @brief Overloads the stream insertion operator for the FinishCounter class.
 *
 * This function allows a description of a FinishCounter to be written to an output
 * stream.
 *
 * @param os The output stream to write to.
 * @param obj The object to write.
 * @return A reference to the modified output stream.
 */
inline std::ostream& operator<<(std::ostream& os, detail::FinishCounter const& obj) {
    os << obj.str();
    return os;
}

}  // namespace rapidsmpf::shuffler
