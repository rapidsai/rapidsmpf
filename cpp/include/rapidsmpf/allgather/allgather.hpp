/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
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

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/buffer/spill_manager.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/statistics.hpp>

/**
 * @namespace rapidsmpf::allgather
 * @brief Allgather communication interfaces.
 *
 * An allgather service for distributed communication where all ranks collect
 * data from all other ranks.
 */
namespace rapidsmpf::allgather {
namespace detail {

/// @brief Type alias for chunk identifiers.
using ChunkID = std::uint64_t;

/**
 * @brief Represents a data chunk in the allgather operation.
 *
 * A chunk is either a data message (in which case metadata indicates
 * how the data are to be interpreted), or a control (finish) message
 * (in which case metadata and data are empty). Chunks within a single
 * `AllGather` operation are uniquely identified by an `(id,
 * is_finish)` pair.
 */
class Chunk {
  private:
    ChunkID id_;  ///< Unique chunk identifier
    std::unique_ptr<std::vector<std::uint8_t>> metadata_;  ///< Serialized metadata
    std::unique_ptr<Buffer> data_;  ///< Data buffer
    std::uint64_t
        data_size_;  ///< Size of data in bytes (maintained separately from the data
                     ///< buffer for validation during `attach_data_buffer`)

    /**
     * @brief Construct a data chunk.
     *
     * @param id Unique chunk identifier.
     * @param metadata Serialized metadata for the chunk.
     * @param data Data buffer containing the chunk's payload.
     */
    Chunk(
        ChunkID id,
        std::unique_ptr<std::vector<std::uint8_t>> metadata,
        std::unique_ptr<Buffer> data
    );

    /**
     * @brief Construct a finish marker chunk.
     *
     * @param id Unique chunk identifier for the finish marker.
     *
     * @note We use the finish marker chunk ID to encode the number of
     * insertions on the originating rank.
     */
    Chunk(ChunkID id);

  public:
    /**
     * @brief Check if the chunk is ready for processing.
     *
     * A chunk is ready either if it has no data buffer, or any
     * outstanding operations on the data buffer have completed.
     *
     * @return True if the chunk is ready, false otherwise.
     */
    [[nodiscard]] bool is_ready() const noexcept;

    /**
     * @brief Return the memory type of the chunk.
     *
     * @return The memory type of the chunk.
     * @note a finish chunk has memory type host.
     */
    [[nodiscard]] MemoryType memory_type() const noexcept;

    /**
     * @brief Check if this is a finish marker chunk.
     *
     * @return True if this chunk represents a finish marker, false otherwise.
     */
    [[nodiscard]] bool is_finish() const noexcept;

    /**
     * @brief The unique identifier of the chunk.
     *
     * @return The chunk's unique identifier.
     */
    [[nodiscard]] ChunkID id() const noexcept;

    /**
     * @brief The sequence number of the chunk.
     *
     * @return The sequence number portion of the chunk ID.
     */
    [[nodiscard]] ChunkID sequence() const noexcept;

    /**
     * @brief The origin rank of the chunk.
     *
     * @return The rank that originated this chunk.
     */
    [[nodiscard]] Rank origin() const noexcept;

    /**
     * @brief The size of the data buffer in bytes.
     *
     * @return The size of the chunk's data buffer.
     */
    [[nodiscard]] std::uint64_t data_size() const noexcept;

    /**
     * @brief The size of the metadata buffer in bytes.
     *
     * @return The size of the chunk's metadata.
     */
    [[nodiscard]] std::uint64_t metadata_size() const noexcept;

    /**
     * @brief Create a data chunk from packed data.
     *
     * @param sequence The sequence number for the chunk.
     * @param origin The originating rank.
     * @param packed_data The packed data to create the chunk from.
     * @return A unique pointer to the created chunk.
     */
    [[nodiscard]] static std::unique_ptr<Chunk> from_packed_data(
        std::uint64_t sequence, Rank origin, PackedData&& packed_data
    );

    /**
     * @brief Create an empty finish marker chunk.
     *
     * @param num_local_insertions The number of data insertions on
     * this rank.
     * @param origin The originating rank.
     * @return A unique pointer to the created finish marker chunk.
     */
    [[nodiscard]] static std::unique_ptr<Chunk> from_empty(
        std::uint64_t num_local_insertions, Rank origin
    );

    /**
     * @brief Release the chunk's data as PackedData.
     *
     * @return The chunk's data and metadata as PackedData.
     *
     * @throws std::logic_error if the chunk is not a data chunk.
     *
     * @note Behaviour is undefined if the chunk is used after being
     * released.
     */
    [[nodiscard]] PackedData release();

    /// @brief Number of bits used for the sequence ID in the chunk identifier.
    static constexpr std::uint64_t ID_BITS = 38;
    /// @brief Number of bits used for the rank in the chunk identifier.
    static constexpr std::uint64_t RANK_BITS =
        sizeof(ChunkID) * std::numeric_limits<unsigned char>::digits - ID_BITS;

    /**
     * @brief Create a `ChunkID` from a sequence number and origin rank.
     *
     * @param sequence the sequence number.
     * @param origin the origin rank.
     *
     * @return The new chunk id.
     */
    static constexpr ChunkID chunk_id(std::uint64_t sequence, Rank origin);

    /**
     * @brief Serialize the metadata of the chunk to a byte vector.
     *
     * @return A vector containing the serialized chunk data.
     */
    [[nodiscard]] std::unique_ptr<std::vector<std::uint8_t>> serialize() const;

    /**
     * @brief Deserialize a chunk from a byte vector of its metadata.
     *
     * @param data The serialized chunk data.
     * @param br Buffer resource for memory allocation.
     * @return A unique pointer to the deserialized chunk.
     *
     * @note If the serialized form encodes a data chunk, this
     * function allocates space for the data buffer.
     */
    [[nodiscard]] static std::unique_ptr<Chunk> deserialize(
        std::vector<std::uint8_t>& data, BufferResource* br
    );

    /**
     * @brief Release and return the data buffer.
     *
     * @return The data buffer, leaving the chunk without data.
     */
    [[nodiscard]] std::unique_ptr<Buffer> release_data_buffer() noexcept;

    /**
     * @brief Attach a data buffer to this chunk.
     *
     * @param data The data buffer to attach.
     * @throws std::logic_error If the `data_size()` of the chunk does
     * not match the size of the provided new data buffer, or the
     * chunk already has a data buffer.
     */
    void attach_data_buffer(std::unique_ptr<Buffer> data);

    /// @brief Default destructor.
    ~Chunk() = default;
    /// @brief Move constructor.
    Chunk(Chunk&&) = default;
    /// @brief Move assignment operator.
    /// @return Moved this
    Chunk& operator=(Chunk&&) = default;
    /// @brief Deleted copy constructor.
    Chunk(Chunk const&) = delete;
    /// @brief Deleted copy assignment operator.
    Chunk& operator=(Chunk const&) = delete;
};

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
     * @param log Logger instance.
     * @param amount Requested amount of data to spill in bytes.
     * @return Actual amount of data spilled in bytes.
     *
     * @note We attempt to minimise the number of individual buffers
     * spilled, as well as the amount of "overspill".
     */
    [[nodiscard]] std::size_t spill(
        BufferResource* br, Communicator::Logger& log, std::size_t amount
    );

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
 * messages ring allgorithms perform better since message passing is
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
     * @param progress_thread The progress thread for asynchronous operations.
     * @param op_id Unique operation identifier for this allgather.
     * @param br Buffer resource for memory allocation.
     * @param statistics Statistics collection instance (disabled by
     * default).
     * @param finished_callback Optional callback run when partitions are locally
     * finished. The callback is guaranteed to be called by the progress thread exactly
     * once when the allgather is locally ready.
     *
     * @note The caller promises that inserted buffers are stream-ordered with respect
     * to their own stream, and extracted buffers are likewise guaranteed to be stream-
     * ordered with respect to their own stream.
     */
    AllGather(
        std::shared_ptr<Communicator> comm,
        std::shared_ptr<ProgressThread> progress_thread,
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
     * @brief Destructor.
     *
     * @note This operation is logically collective. If an `AllGather`
     * is locally destructed before `wait`ing to extract, there is no
     * guarantee that in-flight communication will be completed.
     */
    ~AllGather();

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
    std::shared_ptr<ProgressThread>
        progress_thread_;  ///< Progress thread for async operations
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

}  // namespace rapidsmpf::allgather
