/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/shuffler/postbox.hpp>
#include <rapidsmpf/utils.hpp>

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
 * ranks and partitions. Each rank maintains a counter for tracking how many chunks have
 * been received for each partition.
 */
class FinishCounter {
  public:
    /**
     * @brief Construct a finish counter.
     *
     * @param nranks The total number of ranks participating in the shuffle.
     * @param local_partitions The partition IDs local to the current rank.
     */
    FinishCounter(Rank nranks, std::vector<PartID> const& local_partitions);

    /**
     * @brief Move the goalpost for a specific rank and partition.
     *
     * This function sets the number of chunks that need to be received from a specific
     * rank and partition. It should only be called once per rank and partition.
     *
     * @param pid The partition ID the goalpost is assigned to.
     * @param nchunks The number of chunks required.
     *
     * @throw cudf::logic_error If the goalpost is moved more than once for the same rank
     * and partition.
     */
    void move_goalpost(PartID pid, ChunkID nchunks);

    /**
     * @brief Add a finished chunk to a partition counter.
     *
     * This function increments the finished chunk counter for a specific partition.
     * When the number of finished chunks matches the goalpost, the partition is marked as
     * finished.
     *
     * @param pid The partition ID to update.
     *
     * @throw cudf::logic_error If the partition has already reached the goalpost.
     */
    void add_finished_chunk(PartID pid);

    /**
     * @brief Returns whether all partitions are finished (non-blocking).
     *
     * @return True if all partitions are finished, otherwise false.
     */
    [[nodiscard]] bool all_finished() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return partitions_ready_to_wait_on_.empty();
    }

    /**
     * @brief Returns the partition ID of a finished partition that hasn't been waited on
     * (blocking). Optionally a timeout (in ms) can be provided.
     *
     * This function blocks until a partition is finished and ready to be processed. If
     * the timeout is set and a partition is not available by the time, a
     * std::runtime_error will be thrown.
     *
     * @param timeout Optional timeout (ms) to wait.
     *
     * @return The partition ID of a finished partition.
     *
     * @throw std::out_of_range If all partitions have already been waited on.
     * std::runtime_error If timeout was set and no partitions have been finished by the
     * expiration.
     */
    PartID wait_any(std::optional<std::chrono::milliseconds> timeout = {});

    /**
     * @brief Wait for a specific partition to be finished (blocking). Optionally a
     * timeout (in ms) can be provided.
     *
     * This function blocks until the desired partition is finished and ready
     * to be processed. If the timeout is set and the requested partition is not available
     * by the time, a std::runtime_error will be thrown.
     *
     * @param pid The desired partition ID.
     * @param timeout Optional timeout (ms) to wait.
     *
     * @throw std::out_of_range If the desired partition is unavailable.
     * std::runtime_error If timeout was set and requested partition has been finished by
     * the expiration.
     */
    void wait_on(PartID pid, std::optional<std::chrono::milliseconds> timeout = {});

    /**
     * @brief Returns a vector of partition ids that are finished and haven't been waited
     * on (blocking). Optionally a timeout (in ms) can be provided.
     *
     * This function blocks until at least one partition is finished and ready to be
     * processed. If the timeout is set and no partition is available by the time, a
     * std::runtime_error will be thrown.
     *
     * @param timeout Optional timeout (ms) to wait.
     *
     * @note It is the caller's responsibility to process all returned partition IDs.
     *
     * @return vector of finished partitions.
     *
     * @throw std::out_of_range If all partitions have been waited on.
     * std::runtime_error If timeout was set and no partitions have been finished by the
     * expiration.
     */
    std::vector<PartID> wait_some(std::optional<std::chrono::milliseconds> timeout = {});

    /**
     * @brief Returns a description of this instance.
     * @return The description.
     */
    [[nodiscard]] std::string str() const;


  private:
    Rank const nranks_;
    // The goalpost of each partition. The goal is a rank counter to track how many ranks
    // has reported their goal, and a chunk counter that specifies the goal. It is only
    // when all ranks has reported their goal that the goalpost is final.
    std::unordered_map<PartID, std::pair<Rank, ChunkID>> goalposts_;
    // The finished chunk counter of each partition. The goal of a partition has been
    // reach when its counter equals the goalpost.
    std::unordered_map<PartID, ChunkID> finished_chunk_counters_;
    // A partition has three states:
    //   - If it is false, the partition isn't finished.
    //   - If it is true, the partition is finished and can be waited on.
    //   - If it is absent, the partition is finished and has already been waited on.
    std::unordered_map<PartID, bool> partitions_ready_to_wait_on_;
    mutable std::mutex mutex_;  // TODO: use a shared_mutex lock?
    mutable std::condition_variable cv_;
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
