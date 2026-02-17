/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>
#include <vector>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/packed_data.hpp>

namespace rapidsmpf::shuffler {

/**
 * @brief Partition ID, which goes from 0 to the total number of partitions
 *
 * The `PartID` is always referring to a partition globally.
 */
using PartID = std::uint32_t;

namespace detail {

/**
 * @brief The globally unique ID of a chunk.
 */
using ChunkID = std::uint64_t;

/**
 * @brief A partition chunk representing either a data message or a control message.
 *
 * A Chunk represents a single message for a partition in the shuffler. There are two
 * types.
 *
 * Data Message. Contains actual partition data (metadata and optionally GPU data buffer).
 * - Used to transfer partition data between ranks.
 * - `expected_num_chunks` is 0.
 * - `is_control_message()` returns false.
 * - Contains metadata buffer and optionally a data buffer.
 *
 * Control Message. Signals partition completion without carrying data.
 * - Used to indicate that all data chunks for a partition have been sent.
 * - `expected_num_chunks` > 0 (indicates total number of data chunks expected).
 * - `is_control_message()` returns true.
 * - No metadata or data buffers (metadata_size = 0, data_size = 0).
 *
 * When serialized, the format is:
 * - chunk_id: uint64_t, ID of the chunk.
 * - partition_id: PartID, Partition ID of the message.
 * - expected_num_chunks: size_t, Expected number of chunks (0 for data, >0 for control).
 * - metadata_size: uint32_t, Size of the metadata in bytes.
 * - data_size: uint64_t, Size of the data in bytes.
 * - metadata: vector<uint8_t>, Metadata buffer
 */
class Chunk {
    // friend a method that creates a dummy chunk for testing
    friend Chunk make_dummy_chunk(ChunkID, PartID);

  public:
    /**
     * @brief move constructor
     * @param other The chunk to move.
     */
    Chunk(Chunk&& other) noexcept = default;

    /**
     * @brief move assignment operator
     * @param other The chunk to move.
     * @return this chunk.
     */
    Chunk& operator=(Chunk&& other) noexcept = default;

    // delete copy constructor
    Chunk(Chunk const&) = delete;

    // delete copy assignment operator
    Chunk& operator=(Chunk const&) = delete;

    /**
     * @brief The size of the metadata message header.
     *
     * @return The size of the metadata message header.
     */
    static constexpr size_t metadata_message_header_size() {
        return sizeof(ChunkID) + sizeof(PartID) + sizeof(size_t) + sizeof(uint32_t)
               + sizeof(uint64_t);
    }

    /**
     * @brief The ID of the chunk.
     *
     * @return The ID of the chunk.
     */
    [[nodiscard]] constexpr ChunkID chunk_id() const {
        return chunk_id_;
    }

    /**
     * @brief Partition ID of the message.
     *
     * @return The ID of the partition.
     */
    [[nodiscard]] constexpr PartID part_id() const {
        return part_id_;
    }

    /**
     * @brief The expected number of chunks for the message.
     *
     * @return The expected number of chunks for the message. Non-zero when the message
     * is a control message, otherwise zero (data message).
     */
    [[nodiscard]] constexpr size_t expected_num_chunks() const {
        return expected_num_chunks_;
    }

    /**
     * @brief Whether the message is a control message.
     *
     * @return True if the message is a control message, false otherwise.
     */
    [[nodiscard]] constexpr bool is_control_message() const {
        // We use `expected_num_chunks > 0` to flag a message as a "control message".
        return expected_num_chunks() > 0;
    }

    /**
     * @brief Get the data of the message, as a new chunk.
     *
     * @param new_chunk_id The ID of the new chunk.
     * @param br The buffer resource to use for copying the data.
     * @return A new chunk containing the data of the message.
     *
     * @note This will create a copy of the packed data using a new stream from
     * `br->stream_pool()`. If the message is a data message, the buffers will be moved
     * to the new chunk. If the message is a control message, the metadata and data
     * buffers will be nullptr. For a metadata-only message, the data buffer will be an
     * empty HOST buffer.
     */
    Chunk get_data(ChunkID new_chunk_id, BufferResource* br);

    /**
     * @brief Get the size of the metadata of the message.
     *
     * @return The size of the metadata of the message. Zero when the message is a
     * control message, otherwise the size of `PackedData::metadata`.
     */
    [[nodiscard]] constexpr uint32_t metadata_size() const {
        return metadata_size_;
    }

    /**
     * @brief Get the size of the packed data of the message.
     *
     * @return The size of the packed data of the message. Zero when the message is a
     * control message, otherwise the size of `PackedData::data` of the message.
     */
    [[nodiscard]] constexpr size_t data_size() const {
        return data_size_;
    }

    /**
     * @brief Set the data buffer.
     *
     * @param data The data buffer.
     */
    void set_data_buffer(std::unique_ptr<Buffer> data) {
        RAPIDSMPF_EXPECTS(!data_, "buffer is already set");
        data_ = std::move(data);
    }

    /**
     * @brief Whether the data buffer is set.
     *
     * @return True if the data buffer is set, false otherwise.
     */
    [[nodiscard]] bool is_data_buffer_set() const {
        return data_ != nullptr;
    }

    /**
     * @brief Whether the metadata buffer is set.
     *
     * @return True if the metadata buffer is set, false otherwise.
     */
    [[nodiscard]] bool is_metadata_buffer_set() const {
        return metadata_ != nullptr && !metadata_->empty();
    }

    /**
     * @brief Get the memory type of the data buffer.
     *
     * @return The memory type of the data buffer.
     */
    [[nodiscard]] MemoryType data_memory_type() const {
        RAPIDSMPF_EXPECTS(data_, "data buffer is not set");
        return data_->mem_type();
    }

    /**
     * @brief Release the ownership of the metadata buffer.
     *
     * @return The metadata buffer.
     */
    [[nodiscard]] std::unique_ptr<std::vector<uint8_t>> release_metadata_buffer() {
        return std::move(metadata_);
    }

    /**
     * @brief Release the ownership of the data buffer.
     *
     * @return The data buffer.
     */
    [[nodiscard]] std::unique_ptr<Buffer> release_data_buffer() {
        return std::move(data_);
    }

    /**
     * @brief Create a single-message chunk from a packed data.
     *
     * @param chunk_id The ID of the chunk.
     * @param part_id The ID of the partition.
     * @param packed_data The packed data.
     * @return The chunk.
     */
    static Chunk from_packed_data(
        ChunkID chunk_id, PartID part_id, PackedData&& packed_data
    );

    /**
     * @brief Create a single-message chunk for a finished partition (control
     * message).
     *
     * @param chunk_id The ID of the chunk.
     * @param part_id The ID of the partition.
     * @param expected_num_chunks The expected number of chunks.
     * @return The chunk.
     */
    static Chunk from_finished_partition(
        ChunkID chunk_id, PartID part_id, size_t expected_num_chunks
    );

    /**
     * @brief Create a chunk by deserializing a metadata message.
     *
     * @param msg The metadata message received from another rank.
     * @param validate Whether to validate the metadata buffer.
     * @return The chunk.
     *
     * @throws std::runtime_error if the metadata buffer does not follow the expected
     * format and `validate` is true.
     */
    static Chunk deserialize(std::vector<uint8_t> const& msg, bool validate = true);

    /**
     * @brief Validate if a deserialized buffer follows the Chunk format.
     *
     * @param serialized_buf The deserialized buffer to validate.
     * @return True if the deserialized buffer follows the Chunk format, false
     * otherwise.
     */
    static bool validate_format(std::vector<uint8_t> const& serialized_buf);

    /**
     * @brief Whether the chunk is ready for consumption.
     *
     * @return True if the chunk is ready, false otherwise.
     * @note chunk is ready if it has no data or if the data is ready. data_ buffer
     * could be set later, so we need to check if it is non-null.
     */
    [[nodiscard]] bool is_ready() const {
        // data_size_ contains the size of the data buffer. If it is 0, the chunk
        // has no data, so it is ready. Else, the chunk is ready if the data
        // buffer is non-null and the data buffer is ready.
        return data_size_ == 0 || (data_ && data_->is_latest_write_done());
    }

    /**
     * @brief Returns a description of this chunk.
     *
     * @return The description.
     */
    [[nodiscard]] std::string str() const;

    /**
     * @brief Returns a metadata message that represents this chunk.
     *
     * @returns The metadata message as a serialized byte vector.
     */
    [[nodiscard]] std::unique_ptr<std::vector<uint8_t>> serialize() const;

  private:
    // constructor
    Chunk(
        ChunkID chunk_id,
        PartID part_id,
        size_t expected_num_chunks,
        uint32_t metadata_size,
        uint64_t data_size,
        std::unique_ptr<std::vector<uint8_t>> metadata = nullptr,
        std::unique_ptr<Buffer> data = nullptr
    );

    ChunkID chunk_id_;  ///< The ID of the chunk.
    PartID part_id_;  ///< The partition ID of the message.
    size_t expected_num_chunks_;  ///< The expected number of chunks for the partition.
    uint32_t metadata_size_;  ///< The size of the metadata for the single message.
    uint64_t data_size_;  ///< The size of the data for the single message.

    /// Metadata buffer that contains information about the message in the chunk.
    std::unique_ptr<std::vector<uint8_t>> metadata_;

    /// Data buffer of the message in the chunk.
    std::unique_ptr<Buffer> data_;
};

/**
 * @brief Represents a message indicating readiness to receive data for a specific chunk.
 */
class ReadyForDataMessage {
  public:
    ChunkID cid;  ///< Chunk ID associated with the message.

    /**
     * @brief The size of the message in bytes when serialized.
     *
     * @return The size of the message in bytes.
     */
    static constexpr size_t byte_size = sizeof(ChunkID);

    /**
     * @brief Serializes the message into a byte array.
     *
     * @return A serialized byte vector representing the message.
     */
    [[nodiscard]] std::unique_ptr<std::vector<uint8_t>> pack();

    /**
     * @brief Deserializes a message from a byte array.
     *
     * @param msg A serialized message byte vector.
     * @return A `ReadyForDataMessage` object.
     */
    [[nodiscard]] static ReadyForDataMessage unpack(
        std::unique_ptr<std::vector<uint8_t>> const& msg
    );

    /**
     * @brief Returns a description of this instance.
     * @return The description.
     */
    [[nodiscard]] std::string str() const;
};

/**
 * @brief Overloads the stream insertion operator for the Chunk class.
 *
 * This function allows a description of a Chunk to be written to an output stream.
 *
 * @param os The output stream to write to.
 * @param obj The object to write.
 * @return A reference to the modified output stream.
 */
std::ostream& operator<<(std::ostream& os, Chunk const& obj);

/**
 * @brief Overloads the stream insertion operator for the ReadyForDataMessage class.
 *
 * This function allows a description of a ReadyForDataMessage to be written to an output
 * stream.
 *
 * @param os The output stream to write to.
 * @param obj The object to write.
 * @return A reference to the modified output stream.
 */
std::ostream& operator<<(std::ostream& os, ReadyForDataMessage const& obj);

}  // namespace detail
}  // namespace rapidsmpf::shuffler
