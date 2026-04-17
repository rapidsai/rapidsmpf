/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/coll/utils.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/memory/spill_manager.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/statistics.hpp>

/**
 * @namespace rapidsmpf::coll
 * @brief Collective communication interfaces.
 *
 * An allgather service for distributed communication where all ranks collect
 * data from all other ranks.
 */
namespace rapidsmpf::coll {

/**
 * @brief AllGather communication service.
 *
 * The class provides a communication service where each rank
 * contributes data and all ranks receive all inputs on all ranks.
 *
 * The implementation uses a ring broadcast. Each rank receives a
 * contribution from its left neighbour, forwards the message to its
 * right neighbour (unless at the end of the ring) and then stores the
 * contribution locally. The cost on `P` ranks if each rank inserts a
 * message of size `N` is
 *
 * (P - 1) alpha + N ((P - 1) / P) beta
 *
 * Per insertion. Where alpha is the network latency and beta the
 * inverse bandwidth. Although the latency term is linear (rather than
 * logarithmic as is the case for Bruck's algorithm or recursive
 * doubling) MPI implementations typically observe that for large
 * messages ring algorithms perform better since message passing is
 * only nearest neighbour.
 */
class AllGather {
  public:
    /**
     * @brief Insert packed data into the allgather operation.
     *
     * @param sequence_number Local ordered sequence number of the data.
     * @param packed_data The data to contribute to the allgather.
     */
    void insert(std::uint64_t sequence_number, PackedData&& packed_data);

    /**
     * @brief Mark that this rank has finished contributing data.
     */
    void insert_finished();

    /// @brief Tag requesting ordering for extraction.
    enum class Ordered : bool {
        NO,  ///< Extraction is unordered.
        YES,  ///< Extraction is ordered.
    };

    /**
     * @brief Wait for completion and extract all gathered data.
     *
     * Blocks until the allgather operation completes and returns all
     * collected data from all ranks.
     *
     * @param ordered If the extracted data should be ordered? if
     * ordered, returned data will be ordered first by rank and then by insertion
     * order on that rank.
     * @param timeout Optional maximum duration to wait. Negative values mean no timeout.
     *
     * @return A vector containing packed data from all participating ranks.
     * @throws std::runtime_error If the timeout is reached.
     */
    [[nodiscard]] std::vector<PackedData> wait_and_extract(
        Ordered ordered = Ordered::YES,
        std::chrono::milliseconds timeout = std::chrono::milliseconds{-1}
    );

    /**
     * @brief Construct a new allgather operation.
     *
     * @param comm The communicator for communication.
     * @param op_id Unique operation identifier for this allgather.
     * @param br Buffer resource for memory allocation.
     * @param statistics Statistics collection instance (disabled by
     * default).
     * @param finished_callback Optional callback run when partitions are locally
     * finished. The callback is guaranteed to be called by the progress thread exactly
     * once when the allgather is locally ready.
     *
     * @note It is safe to reuse the `op_id` as soon as `wait_and_extract` has completed
     * locally.
     *
     * @note The caller promises that inserted buffers are stream-ordered with respect
     * to their own stream, and extracted buffers are likewise guaranteed to be stream-
     * ordered with respect to their own stream.
     */
    AllGather(
        std::shared_ptr<Communicator> comm,
        OpID op_id,
        BufferResource* br,
        std::shared_ptr<Statistics> statistics = Statistics::disabled(),
        std::function<void(void)>&& finished_callback = nullptr
    );

    /// @brief Deleted copy constructor.
    AllGather(AllGather const&) = delete;
    /// @brief Deleted copy assignment operator.
    AllGather& operator=(AllGather const&) = delete;
    /// @brief Deleted move constructor.
    AllGather(AllGather&&) = delete;
    /// @brief Deleted move assignment operator.
    AllGather& operator=(AllGather&&) = delete;

    /**
     * @brief Gets the communicator associated with this AllGather.
     *
     * @return Shared pointer to communicator.
     */
    [[nodiscard]] std::shared_ptr<Communicator> const& comm() const noexcept {
        return comm_;
    }

    /**
     * @brief Destructor.
     *
     * @note This operation is logically collective. If an `AllGather`
     * is locally destructed before `wait`ing to extract, there is no
     * guarantee that in-flight communication will be completed.
     */
    ~AllGather() noexcept;

    /**
     * @brief Main event loop for processing allgather operations.
     *
     * This method is called by the progress thread to handle ongoing
     * communication and data transfers.
     *
     * @return The current progress state.
     */
    ProgressThread::ProgressState event_loop();

  private:
    /**
     * @brief Insert a chunk directly into the allgather operation.
     *
     * @param chunk The chunk to insert.
     */
    void insert(std::unique_ptr<detail::Chunk> chunk);

    /**
     * @brief Handle a finish message.
     *
     * @param expected_chunks The expected number of chunks we expect
     * from the rank this finish message is from.
     */
    void mark_finish(std::uint64_t expected_chunks) noexcept;

    /**
     * @brief Wait for the allgather operation to complete.
     *
     * @param timeout Optional maximum duration to wait. Negative values mean no timeout.
     *
     * @throws std::runtime_error If the timeout is reached.
     */
    void wait(std::chrono::milliseconds timeout = std::chrono::milliseconds{-1});

    /**
     * @brief Attempt to spill device memory
     *
     * @param amount Optional amount of memory to spill.
     *
     * @return The amount of memory actually spilled.
     */
    std::size_t spill(std::optional<std::size_t> amount = std::nullopt);

    std::shared_ptr<Communicator> comm_;  ///< Communicator
    BufferResource* br_;  ///< Buffer resource for memory allocation
    std::shared_ptr<Statistics> statistics_;  ///< Statistics collection instance
    std::function<void(void)> finished_callback_{
        nullptr
    };  ///< Optional callback to run when allgather is finished and ready for extraction.
    std::atomic<Rank> finish_counter_;  ///< Counter for finish markers received
    std::atomic<std::uint32_t> nlocal_insertions_;  ///< Number of local data insertions
    std::atomic<std::uint64_t> extraction_goalpost_{
        0
    };  ///< Number of chunks still expected to remain in the extraction postbox.
    OpID op_id_;  ///< Unique operation identifier
    std::atomic<bool> locally_finished_{false};  ///< Whether this rank has finished
    bool can_extract_{false};  ///< Whether data can be extracted
    mutable std::mutex mutex_;  ///< Mutex protecting can_extract_
    std::condition_variable cv_;  ///< Notification for waiting on can_extract_
    detail::PostBox inserted_{};  ///< Postbox for chunks inserted by user/event loop
    detail::PostBox for_extraction_{};  ///< Postbox for chunks ready for user extraction
    ProgressThread::FunctionID function_id_{};  ///< Function ID in progress thread
    SpillManager::SpillFunctionID spill_function_id_{};  ///< Function ID for spilling
    // We track remote finishes separately from the finish_counter_ above since the path
    // through the event loop state machine for a local finish marker is slightly
    // different from a remote finish marker.
    /// @brief Number of remote finish messages received.
    Rank remote_finish_counter_;
    /// @brief Total expected data-carrying messages.
    std::uint64_t num_expected_messages_{0};
    /// @brief Total data-carrying messages messages received so far.
    std::uint64_t num_received_messages_{0};
    /// @brief Chunks being received from left neighbor
    std::vector<std::unique_ptr<detail::Chunk>> to_receive_{};
    /// @brief Fire-and-forget communication futures
    std::vector<std::unique_ptr<Communicator::Future>> fire_and_forget_{};
    /// @brief Chunks for which a send future is posted
    std::vector<std::unique_ptr<detail::Chunk>> sent_posted_{};
    /// @brief Futures for posted sends. Order matches
    std::vector<std::unique_ptr<Communicator::Future>> sent_futures_{};
    /// @brief Chunks for which a receive future is posted
    std::vector<std::unique_ptr<detail::Chunk>> receive_posted_{};
    /// @brief Futures for posted receives. Order matches.
    std::vector<std::unique_ptr<Communicator::Future>> receive_futures_{};
};

}  // namespace rapidsmpf::coll
