/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <functional>
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
     * @brief Callback function type called when a partition is finished.
     *
     * The callback receives the partition ID of the finished partition.
     *
     * @warning A callback must be fast and non-blocking and should not call any of the
     * `wait*` methods. And be very careful if acquiring locks. Ideally it should be used
     * to signal a separate thread to do the actual processing (eg. WaitHand).
     *
     * @note When a callback is registered, it will be identified by the
     * FinishedCbId returned. So, if a callback needs to be preemptively canceled,
     * the corresponding identifier needs to be provided.
     *
     * @note Every callback will be called as and when each partition is finished. If
     * there were finished partitions before the callback was registered, the callback
     * will be called for them immediately by the caller thread. Else, the callback will
     * be called by the progress thread (Therefore, it will be called
     * `n_local_partitions_` times in total).
     *
     * @note Caller needs to be careful when using both callbacks and wait* methods
     * together.
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
        std::vector<PartID> const& local_partitions,
        FinishedCallback&& finished_callback = nullptr
    );

    ~FinishCounter();

    /**
     * @brief Move the goalpost for a specific rank and partition.
     *
     * This function sets the number of chunks that need to be received from a specific
     * rank and partition. It should only be called once per rank and partition.
     *
     * @param pid The partition ID the goalpost is assigned to.
     * @param nchunks The number of chunks required. (Requires nchunks > 0)
     *
     * @throws std::logic_error If the goalpost is moved more than once for the same rank
     * and partition, or if nchunks is 0.
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
     * @throws std::logic_error If the partition has already reached the goalpost.
     */
    void add_finished_chunk(PartID pid);

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
     * This function blocks until a partition is finished and ready to be processed. If
     * the timeout is set and a partition is not available by the time, a
     * std::runtime_error will be thrown.
     *
     * @param timeout Optional timeout (ms) to wait.
     *
     * @return The partition ID of a finished partition.
     *
     * @throws std::runtime_error If all partitions have already been waited on or if
     * timeout was set and no partitions have been finished by the expiration.
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
     * @throws std::runtime_error If the desired partition is unavailable or if timeout
     * was set and requested partition has not been finished by the expiration.
     */
    void wait_on(PartID pid, std::optional<std::chrono::milliseconds> timeout = {});

    /**
     * @brief Returns a description of this instance.
     * @return The description.
     */
    [[nodiscard]] std::string str() const;

  private:
    Rank const nranks_;

    /// @brief Information about a local partition.
    struct PartitionInfo {
        Rank rank_count{0};  ///< number of ranks that have reported their chunk count.
        ChunkID chunk_goal{0};  ///< the goal of a partition. This keeps increasing until
                                ///< all ranks have reported their chunk count.
        ChunkID finished_chunk_count{
            0
        };  ///< The finished chunk counter of each partition. The goal of a partition has
        ///< been reached when its counter equals the goalpost.

        constexpr PartitionInfo() = default;

        constexpr void move_goalpost(ChunkID nchunks, Rank nranks) {
            RAPIDSMPF_EXPECTS(nchunks != 0, "the goalpost was moved by 0 chunks");
            RAPIDSMPF_EXPECTS(
                ++rank_count <= nranks, "the goalpost was moved more than one per rank"
            );
            chunk_goal += nchunks;
        }

        constexpr void add_finished_chunk(Rank nranks) {
            finished_chunk_count++;
            // only throw if rank_count == nranks
            RAPIDSMPF_EXPECTS(
                (rank_count < nranks) || (finished_chunk_count <= chunk_goal),
                "finished chunk exceeds the goal"
            );
        }

        // The partition is finished if the goalpost has been set by all ranks
        // and the number of finished chunks has reached the goal.
        [[nodiscard]] constexpr bool is_finished(Rank nranks) const {
            return rank_count == nranks && finished_chunk_count == chunk_goal;
        }

        [[nodiscard]] constexpr ChunkID data_chunk_goal() const {
            // there will always be a control message from each rank indicating how many
            // chunks it's sending. Chunk goal contains this control message for each
            // rank. Therefore, to get the data chunk goal, we need to subtract the number
            // of ranks that have reported their chunk count from the chunk goal.
            return chunk_goal - static_cast<ChunkID>(rank_count);
        }
    };

    // The goalpost of each partition. The goal is a rank counter to track how many ranks
    // has reported their goal, and a chunk counter that specifies the goal. It is only
    // when all ranks has reported their goal that the goalpost is final.
    std::unordered_map<PartID, PartitionInfo> goalposts_;

    // mutex to control access between the progress thread and the caller thread on shared
    // resources
    mutable std::mutex mutex_;  // TODO: use a shared_mutex lock?

    ///@brief Handler to implement the wait* methods using callbacks
    struct WaitHandler {
        std::unordered_set<PartID>
            to_wait{};  ///< finished partitions available to wait on
        bool active{true};
        std::condition_variable cv;
        std::mutex mutex;

        ~WaitHandler();

        // Callback to listen on the finished partitions
        void on_finished_cb(PartID pid);

        PartID wait_any(std::optional<std::chrono::milliseconds> timeout);

        void wait_on(PartID pid, std::optional<std::chrono::milliseconds> timeout);
    };

    WaitHandler wait_handler_{};

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
