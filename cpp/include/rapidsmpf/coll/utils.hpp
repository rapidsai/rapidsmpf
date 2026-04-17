/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>

namespace rapidsmpf::coll::detail {

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
    Rank destination_;  ///< Destination rank
    std::unique_ptr<std::vector<std::uint8_t>> metadata_;  ///< Serialized metadata
    std::unique_ptr<Buffer> data_;  ///< Data buffer
    std::uint64_t
        data_size_;  ///< Size of data in bytes (maintained separately from the data
                     ///< buffer for validation during `attach_data_buffer`)

    /**
     * @brief Construct a data chunk.
     *
     * @param id Unique chunk identifier.
     * @param destination Destination rank, only used on send side, or `INVALID_RANK` if
     * unset.
     * @param metadata Serialized metadata for the chunk.
     * @param data Data buffer containing the chunk's payload.
     *
     * @throw std::invalid_argument If either @p metadata and @p data are null.
     * @throw std::logic_error If one of @p metadata or @p data are null, but not both
     * (see warning for details).
     *
     * @warning The caller is responsible to ensure both metadata and data are non-null,
     * providing only one leads to an irrecoverable condition.
     */
    Chunk(
        ChunkID id,
        Rank destination,
        std::unique_ptr<std::vector<std::uint8_t>> metadata,
        std::unique_ptr<Buffer> data
    );

    /**
     * @brief Construct a finish marker chunk.
     *
     * @param id Unique chunk identifier for the finish marker.
     * @param destination Destination rank, or `INVALID_RANK` if unset.
     *
     * @note We use the finish marker chunk ID to encode the number of
     * insertions on the originating rank.
     */
    Chunk(ChunkID id, Rank destination);

  public:
    /// @brief Sentinel destination for chunks with unknown destination.
    ///
    /// A received chunk will have an invalid destination.
    static constexpr Rank INVALID_RANK = std::numeric_limits<Rank>::max();

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
     * @brief The local destination rank associated with this chunk.
     *
     * This value is not serialized. Chunks reconstructed from the wire
     * format are assigned `INVALID_RANK`.
     *
     * @return The destination rank, or `INVALID_RANK` if unset.
     */
    [[nodiscard]] Rank destination() const noexcept;

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
     * @param destination The destination rank.
     * @param packed_data The packed data to create the chunk from.
     * @return A unique pointer to the created chunk.
     */
    [[nodiscard]] static std::unique_ptr<Chunk> from_packed_data(
        std::uint64_t sequence, Rank origin, Rank destination, PackedData&& packed_data
    );

    /**
     * @brief Create an empty finish marker chunk.
     *
     * @param num_local_insertions The number of data insertions on
     * this rank.
     * @param origin The originating rank.
     * @param destination The destination rank.
     * @return A unique pointer to the created finish marker chunk.
     */
    [[nodiscard]] static std::unique_ptr<Chunk> from_empty(
        std::uint64_t num_local_insertions, Rank origin, Rank destination
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
 * @brief A thread-safe container for managing chunks in collectives.
 *
 * A `PostBox` provides a synchronized storage mechanism for chunks, allowing
 * multiple threads to insert chunks and extract ready chunks safely.
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
     * @brief Check the number of chunks currently stored.
     *
     * @return The number of chunks currently in the postbox.
     */
    [[nodiscard]] std::size_t size() const noexcept;

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
};

/**
 * @brief Complete posted chunk receives.
 *
 * Tests a set of outstanding communicator futures, and for every
 * completed receive, reattaches the received data buffer to the
 * corresponding chunk before returning it.
 *
 * @param chunks Chunks with receives currently in flight.
 * @param futures Receive futures corresponding 1:1 with @p chunks.
 * @param comm Communicator used to test and release completed receives.
 * @return The subset of chunks whose receives completed during this call.
 */
[[nodiscard]] std::vector<std::unique_ptr<Chunk>> test_some(
    std::vector<std::unique_ptr<Chunk>>& chunks,
    std::vector<std::unique_ptr<Communicator::Future>>& futures,
    Communicator* comm
);

}  // namespace rapidsmpf::coll::detail
