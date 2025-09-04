/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
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
     * @brief Returns whether a partition is finished (non-blocking).
     *
     * @param pid The partition ID to check.
     * @return True if the partition is finished, otherwise False.
     */
    [[nodiscard]] bool is_finished(PartID pid) const;

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
     * @brief Type used to identify callbacks.
     */
    using FinishedCbId = size_t;

    /**
     * @brief Register a callback to be notified when any partition is finished.
     *
     * This function registers a callback that will be called when a partition is finished
     * (and for all currently finished partitions). The callback receives partition IDs as
     * they complete. If all partitions are already finished, the callback is executed
     * immediately for all partitions and invalid_cb_id is returned.
     *
     * @param cb The callback to invoke when partitions are finished.
     *
     * @return A unique callback ID that can be used to cancel the callback, or
     * invalid_cb_id if the callback was executed immediately.
     */
    FinishedCbId register_finished_callback(FinishedCallback&& cb);

    /**
     * @brief Special constant indicating an invalid or immediately-executed callback ID.
     *
     * This value is returned by register_finished_callback when the callback is executed
     * immediately (e.g., when all partitions are already finished).
     */
    static constexpr FinishedCbId invalid_cb_id =
        std::numeric_limits<FinishedCbId>::max();

    /**
     * @brief Cancel a previously registered callback.
     *
     * This function removes a callback registered with register_finished_callback using
     * its ID. It is safe to call this with invalid_cb_id or an already-cancelled ID.
     *
     * @param callback_id callback ID.
     */
    void remove_finished_callback(FinishedCbId callback_id);

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

    std::vector<PartID> finished_partitions_{};  ///< partition IDs of finished partitions

    std::mutex finished_cbs_mutex_;  ///< mutex to protect the finished_cbs_ and
                                     ///< next_finished_cb_id_

    struct CallbackContainer {
        FinishedCbId cb_id;  ///< callback ID to identify the callback

        // index of the next partition that the callback is interested in. cb will
        // called from next_pid_idx to end of finished_partitions_
        size_t next_pid_idx;

        FinishedCallback cb;
    };

    std::vector<CallbackContainer> finished_cbs_{};
    FinishedCbId next_finished_cb_id_{0};  ///< next callback ID to assign

    // mutex to control access between the progress thread and the caller thread on shared
    // resources
    mutable std::mutex mutex_;  // TODO: use a shared_mutex lock?

    class WaitHandler;  ///< Handler to implement the wait* methods using callbacks
    std::unique_ptr<WaitHandler> wait_handler_;
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
