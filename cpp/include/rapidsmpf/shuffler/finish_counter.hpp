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

    /**
     * @brief Move the goalpost for a specific rank and partition.
     *
     * This function sets the number of chunks that need to be received from a specific
     * rank and partition. It should only be called once per rank and partition.
     *
     * @param pid The partition ID the goalpost is assigned to.
     * @param nchunks The number of chunks required. (Requires nchunks > 0)
     *
     * @throw std::logic_error If the goalpost is moved more than once for the same rank
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
     * @throw std::logic_error If the partition has already reached the goalpost.
     */
    void add_finished_chunk(PartID pid);

    /**
     * @brief Returns whether all partitions are finished (non-blocking).
     *
     * @return True if all partitions are finished, otherwise False.
     */
    [[nodiscard]] bool all_finished() const;

    /**
     * @brief Callback function type called when a specific partition is finished.
     *
     * The callback receives a boolean indicating whether the partition contains data.
     */
    using FinishedCallback = std::function<void(bool)>;

    /**
     * @brief Register a callback to be notified when a specific partition is finished.
     *
     * This function registers a callback that will be called when the specified partition
     * completes all its chunks. Only one callback can be registered per partition.
     *
     * @param pid The partition ID to monitor.
     * @param cb The callback to invoke when the partition is finished.
     *
     * @note The callback will be called when the partition is finished and ready to be
     * processed. If the partition is already finished and ready, the callback will be
     * called immediately. If the partition is not finished, the callback will be called
     * when the partition is finished and ready to be processed.
     *
     * @throw std::logic_error If a callback is already registered for this partition.
     */
    void on_finished(PartID pid, FinishedCallback cb);

    /**
     * @brief Cancel a previously registered callback for a specific partition.
     *
     * This function removes any callback registered for the specified partition.
     * It is safe to call this even if no callback is registered.
     *
     * @param pid The partition ID whose callback should be cancelled.
     */
    void cancel_finished_callback(PartID pid);

    /**
     * @brief Callback function type called when any partition is finished.
     *
     * The callback receives the partition ID and a boolean indicating whether
     * the partition contains data.
     */
    using FinishedAnyCallback = std::function<void(PartID, bool)>;

    /**
     * @brief Type used to identify callbacks registered with on_finished_any.
     */
    using FinishedCbId = size_t;

    /**
     * @brief Special constant indicating an invalid or immediately-executed callback ID.
     *
     * This value is returned by on_finished_any when the callback is executed
     * immediately (e.g., when a partition is already ready).
     */
    static constexpr FinishedCbId invalid_cb_id =
        std::numeric_limits<FinishedCbId>::max();

    /**
     * @brief Register a callback to be notified when any partition is finished.
     *
     * This function registers a callback that will be called when any partition
     * completes. If a partition is already finished and ready, the callback may
     * be executed immediately.
     *
     * @param cb The callback to invoke when any partition is finished.
     *
     * @return A callback ID that can be used to cancel the callback, or invalid_cb_id
     *         if the callback was executed immediately.
     *
     * @note The callback will be called when the partition is finished and ready to be
     * processed. If the partition is already finished and ready, the callback will be
     * called immediately. If the partition is not finished, the callback will be called
     * when the partition is finished and ready to be processed.
     *
     * @throw std::logic_error If all partitions are already finished.
     */
    FinishedCbId on_finished_any(FinishedAnyCallback cb);

    /**
     * @brief Cancel a previously registered callback for any partition completion.
     *
     * This function removes a callback registered with on_finished_any using its ID.
     * It is safe to call this with invalid_cb_id or an already-cancelled ID.
     *
     * @param callback_id The ID returned by on_finished_any to cancel.
     */
    void cancel_finished_any_callback(FinishedCbId callback_id);

    /**
     * @brief RAII guard for automatic callback cleanup.
     *
     * CallbackGuard provides automatic cleanup of registered callbacks when the guard
     * goes out of scope. This ensures that callbacks are properly cleaned up even if
     * exceptions occur or timeouts happen. The guard is non-copyable and non-movable
     * to prevent double cleanup and ensure unique ownership.
     *
     * @tparam T Either PartID for partition-specific callbacks or FinishedCbId for
     *           any-partition callbacks.
     */
    template <typename T>
        requires std::same_as<T, PartID> || std::same_as<T, FinishedCbId>
    struct CallbackGuard;

    /**
     * @brief Register a callback for a specific partition with automatic cleanup.
     *
     * This function combines on_finished() with automatic cleanup via RAII. The returned
     * guard will automatically cancel the callback when it goes out of scope, ensuring
     * proper cleanup even in timeout or exception scenarios.
     *
     * @param pid The partition ID to monitor.
     * @param cb The callback to invoke when the partition is finished (moved into the guard).
     *
     * @return A CallbackGuard that will automatically cleanup the callback.
     *
     * @throw std::logic_error If a callback is already registered for this partition.
     *
     * @note The returned guard must be stored in a variable to ensure proper lifetime
     *       management. Discarding the return value will cause immediate cleanup.
     * @note The callback must be passed as an rvalue (use std::move() for lvalues).
     */
    CallbackGuard<PartID> on_finished_with_guard(PartID pid, FinishedCallback&& cb);

    /**
     * @brief Register a callback for any partition completion with automatic cleanup.
     *
     * This function combines on_finished_any() with automatic cleanup via RAII. The
     * returned guard will automatically cancel the callback when it goes out of scope,
     * ensuring proper cleanup even in timeout or exception scenarios.
     *
     * @param cb The callback to invoke when any partition is finished (moved into the guard).
     *
     * @return A CallbackGuard that will automatically cleanup the callback.
     *
     * @throw std::logic_error If all partitions are already finished.
     *
     * @note The returned guard must be stored in a variable to ensure proper lifetime
     *       management. Discarding the return value will cause immediate cleanup.
     * @note The callback must be passed as an rvalue (use std::move() for lvalues).
     */
    CallbackGuard<FinishedCbId> on_finished_any_with_guard(FinishedAnyCallback&& cb);

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
     * @return The partition ID of a finished partition and a boolean indicating if the
     * partition contains data.
     *
     * @throw std::out_of_range If all partitions have already been waited on.
     * std::runtime_error If timeout was set and no partitions have been finished by the
     * expiration.
     */
    std::pair<PartID, bool> wait_any(
        std::optional<std::chrono::milliseconds> timeout = {}
    );

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
     * @return A boolean indicating if the partition contains data.
     *
     * @throw std::out_of_range If the desired partition is unavailable.
     * std::runtime_error If timeout was set and requested partition has been finished by
     * the expiration.
     */
    bool wait_on(PartID pid, std::optional<std::chrono::milliseconds> timeout = {});

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
     * @return A pair of vectors of finished partitions and a boolean indicating if the
     * partition contains data for each partition.
     *
     * @throw std::out_of_range If all partitions have been waited on.
     * std::runtime_error If timeout was set and no partitions have been finished by the
     * expiration.
     */
    std::pair<std::vector<PartID>, std::vector<bool>> wait_some(
        std::optional<std::chrono::milliseconds> timeout = {}
    );

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
        ChunkID finished_chunk_count{0
        };  ///< The finished chunk counter of each partition. The goal of a partition has
        ///< been reached when its counter equals the goalpost.

        FinishedCallback finished_cb{
        };  ///< callback to notify when the partition is finished

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

    std::unordered_map<PartID, bool> ready_pids_stash_{
    };  ///< partition IDs of ready partitions

    std::unordered_map<FinishedCbId, FinishedAnyCallback> finished_any_cbs_{
    };  ///< callbacks to notify when any partition is finished
    FinishedCbId next_finished_cb_id_{0};

    mutable std::mutex mutex_;  // TODO: use a shared_mutex lock?

    std::mutex wait_mutex_;  ///< mutex to protect the wait_cv_
    std::condition_variable wait_cv_;  ///< condition variable to wait on by wait* methods
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
