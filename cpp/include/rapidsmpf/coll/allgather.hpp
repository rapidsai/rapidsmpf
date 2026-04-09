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
#include <limits>
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
namespace detail {

/**
 * @brief A thread-safe container for managing chunks in an `AllGather`.
 *
 * A `PostBox` provides a synchronized storage mechanism for chunks, allowing
 * multiple threads to insert chunks and extract ready chunks safely.
 * It maintains a goalpost mechanism to track when all expected chunks
 * have been received.
 */
class PostBox {
  public:
    /// @brief Default constructor.
    PostBox() = default;
    /// @brief Default destructor.
    ~PostBox() = default;
    /// @brief Deleted copy constructor.
    PostBox(PostBox const&) = delete;
    /// @brief Deleted copy assignment operator.
    PostBox& operator=(PostBox const&) = delete;
    /// @brief Deleted move constructor.
    PostBox(PostBox&&) = delete;
    /// @brief Deleted move assignment operator.
    PostBox& operator=(PostBox&&) = delete;

    /**
     * @brief Insert a single chunk into the postbox.
     *
     * @param chunk The chunk to insert.
     */
    void insert(std::unique_ptr<Chunk> chunk);

    /**
     * @brief Insert multiple chunks into the postbox.
     *
     * @param chunks A vector of chunks to insert.
     */
    void insert(std::vector<std::unique_ptr<Chunk>>&& chunks);

    /**
     * @brief Increment the goalpost to a new expected chunk count.
     *
     * @param amount The amount to move the goalpost by.
     */
    void increment_goalpost(std::uint64_t amount);

    /**
     * @brief Check if the postbox has reached its goal.
     *
     * @return True if the number of stored chunks matches the current
     * goalpost, false otherwise.
     */
    [[nodiscard]] bool ready() const noexcept;

    /**
     * @brief Extract ready chunks from the postbox.
     *
     * @return A vector of chunks that are ready for processing.
     *
     * @note Ready chunks are those with no pending operations on
     * their data buffers.
     */
    [[nodiscard]] std::vector<std::unique_ptr<Chunk>> extract_ready();

    /**
     * @brief Extract all chunks from the postbox.
     *
     * @return A vector containing all chunks in the postbox.
     *
     * @note The caller must ensure that any subsequent operations on
     * the return chunks are stream-ordered.
     */
    [[nodiscard]] std::vector<std::unique_ptr<Chunk>> extract();

    /**
     * @brief Check if the postbox is empty.
     *
     * @return True if the postbox contains no chunks, false otherwise.
     */
    [[nodiscard]] bool empty() const noexcept;

    /**
     * @brief Spill device data from the post box.
     *
     * The spilling is stream ordered by the spilled buffers' CUDA streams.
     *
     * @param br The buffer resource for host and device allocations.
     * @param amount Requested amount of data to spill in bytes.
     * @return Actual amount of data spilled in bytes.
     *
     * @note We attempt to minimise the number of individual buffers
     * spilled, as well as the amount of "overspill".
     */
    [[nodiscard]] std::size_t spill(BufferResource* br, std::size_t amount);

  private:
    mutable std::mutex mutex_{};  ///< Mutex for thread-safe access
    std::vector<std::unique_ptr<Chunk>> chunks_{};  ///< Container for stored chunks
    std::atomic<std::uint64_t> goalpost_{0};  ///< Expected number of chunks
};

}  // namespace detail

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

    /**
     * @brief Check if the allgather operation has completed.
     *
     * @return True if we have received all data and finish messages from all
     * ranks.
     */
    [[nodiscard]] bool finished() const noexcept;

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
     * @brief Extract any available partitions.
     *
     * @return A vector containing available data (or empty if none).
     *
     * @note This is a non-blocking, unordered interface.
     *
     * Example usage to drain an `AllGather`:
     * @code{.cpp}
     * auto allgather = ...; // create
     * ...; // insert data
     * allgather->insert_finished(); // finish inserting
     * std::vector<PackedData> results;
     * while (!allgather->finished()) {
     *    std::ranges::move(allgather->extract_ready(), std::back_inserter(results));
     * }
     * // Extract any final chunks that may have arrived.
     * std::ranges::move(allgather->extract_ready(), std::back_inserter(results));
     * @endcode
     */
    [[nodiscard]] std::vector<PackedData> extract_ready();

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
    OpID op_id_;  ///< Unique operation identifier
    std::atomic<bool> locally_finished_{false};  ///< Whether this rank has finished
    std::atomic<bool> active_{true};  ///< Whether the operation is active
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
