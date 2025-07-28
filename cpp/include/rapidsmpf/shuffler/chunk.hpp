/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>
#include <sstream>
#include <vector>

#include <cuda/std/span>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/communicator/communicator.hpp>

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
 * @brief Chunk with multiple messages. This class contains two buffers for concatenated
 * metadata and data.
 *
 * When the Chunk is serialized, the format is as follows:
 * - chunk_id: uint64_t, ID of the chunk
 * - n_elements: size_t, Number of messages in the chunk
 * - [partition_ids]: vector<PartID>, Partition IDs of the messages, size = n_elements
 * - [expected_num_chunks]: vector<size_t>, Expected number of chunks of the messages,
 * size = n_elements
 * - [meta_offsets]: vector<uint32_t>, Prefix sums (excluding 0) of the metadata sizes
 * of the messages, size = n_elements
 * - [data_offsets]: vector<uint64_t>, Prefix sums (excluding 0) of the data sizes of
 * the messages, size = n_elements
 * - [concat_metadata]: vector<uint8_t>, Concatenated metadata of the messages,
 * size = meta_offsets[n_elements - 1]
 *
 * For a chunk with N messages with M bytes of concat metadata the size of serialized
 * buffer is sizeof(ChunkID) + sizeof(size_t) + N * (sizeof(PartID) + sizeof(size_t) +
 * sizeof(uint32_t) + sizeof(uint64_t)) + M = 16 + N * 24 + M bytes.
 *
 * For a chunk with a single control message (ie. M = 0), the size of the serialized
 * buffer is 40 bytes.
 *
 * For a chunk with a single message with M bytes of metadata, the size of the serialized
 * buffer is 40 + M bytes.
 */
class Chunk {
    // friend a method that creates a dummy chunk for testing
    friend Chunk make_dummy_chunk(ChunkID, PartID);

  public:
    /**
     * @brief The size of the metadata message header.
     *
     * @param n_messages The number of messages in the chunk.
     * @return The size of the metadata message header.
     */
    static constexpr size_t metadata_message_header_size(size_t n_messages) {
        return sizeof(ChunkID) + sizeof(size_t)
               + n_messages
                     * (sizeof(PartID) + sizeof(size_t) + sizeof(uint32_t)
                        + sizeof(uint64_t));
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
     * @brief The number of messages in the chunk.
     *
     * @return The number of messages in the chunk.
     */
    [[nodiscard]] size_t n_messages() const {
        return part_ids_.size();
    }

    /**
     * @brief Partition ID of the i-th message.
     *
     * @param i The index of the message.
     * @return The ID of the partition.
     */
    [[nodiscard]] PartID part_id(size_t i) const {
        return part_ids_.at(i);
    }

    /**
     * @brief The expected number of chunks of the i-th message.
     *
     * @param i The index of the message.
     * @return The expected number of chunks for the message. Non-zero when the message
     * is a control message, otherwise zero (data message).
     */
    [[nodiscard]] size_t expected_num_chunks(size_t i) const {
        return expected_num_chunks_.at(i);
    }

    /**
     * @brief Whether the i-th message is a control message.
     *
     * @param i The index of the message.
     * @return True if the message is a control message, false otherwise.
     */
    [[nodiscard]] inline bool is_control_message(size_t i) const {
        // We use `expected_num_chunks > 0` to flag a message as a "control message".
        return expected_num_chunks(i) > 0;
    }

    /**
     * @brief Get the data of the i-th message, as a new chunk.
     *
     * @param new_chunk_id The ID of the new chunk.
     * @param i The index of the message.
     * @param stream The CUDA stream to use for copying the data.
     * @param br The buffer resource to use for copying the data.
     * @return A new chunk containing the data of the i-th message.
     * @note This will create a copy of the packed data. If there is only one message and
     * the message is a data message, the buffers will be moved to the new chunk.
     * Otherwise a new chunk will be created by copying data. If the i'th message is,
     *  - control message, the metadata and data buffers will be nullptr
     *  - data message, both metadata and data buffers will be non-null (for a
     *    metadata-only message, the data buffer will be an empty HOST buffer)
     *
     * @throws std::out_of_range if the index is out of bounds.
     */
    Chunk get_data(
        ChunkID new_chunk_id, size_t i, rmm::cuda_stream_view stream, BufferResource* br
    );

    /**
     * @brief Get the size of the metadata of the i-th message.
     *
     * @param i The index of the message.
     * @return The size of the metadata of the message. Zero when the message is a
     * control message, otherwise the size of `PackedData::metadata`.
     */
    [[nodiscard]] uint32_t metadata_size(size_t i) const {
        return i == 0 ? meta_offsets_.at(0)
                      : meta_offsets_.at(i) - meta_offsets_.at(i - 1);
    }

    /**
     * @brief Get the size of the packed data of the i-th message.
     *
     * @param i The index of the message.
     * @return The size of the packed data of the message. Zero when the message is a
     * control message, otherwise the size of `PackedData::gpu_data` of the message.
     */
    [[nodiscard]] size_t data_size(size_t i) const {
        return i == 0 ? data_offsets_.at(0)
                      : data_offsets_.at(i) - data_offsets_.at(i - 1);
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
     * @brief Get the size of the concatenated data.
     *
     * @return The size of the concatenated data.
     */
    [[nodiscard]] size_t concat_data_size() const {
        return data_offsets_[n_messages() - 1];
    }

    /**
     * @brief Get the size of the concatenated metadata.
     *
     * @return The size of the concatenated metadata.
     */
    [[nodiscard]] size_t concat_metadata_size() const {
        return meta_offsets_[n_messages() - 1];
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
     * @param event The CUDA event.
     * @param stream The CUDA stream.
     * @param br The buffer resource.
     * @return The chunk.
     */
    static Chunk from_packed_data(
        ChunkID chunk_id,
        PartID part_id,
        PackedData&& packed_data,
        std::shared_ptr<Buffer::Event> event,
        rmm::cuda_stream_view stream,
        BufferResource* br
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
    [[nodiscard]] inline bool is_ready() const {
        // data_offsets_[-1] contains the size of the data buffer. If it is 0, the chunk
        // has no data messages, so it is ready. Else, the chunk is ready if the data
        // buffer is non-null and the data buffer is ready.
        return !data_offsets_.empty()
               && (data_offsets_[n_messages() - 1] == 0 || (data_ && data_->is_ready()));
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

    /**
     * @brief Concatenate multiple chunks into a single chunk.
     *
     * @param chunks Vector of chunks to concatenate. The chunks will be moved from this
     * vector.
     * @param chunk_id The ID for the resulting concatenated chunk.
     * @param stream The CUDA stream to use for copying data.
     * @param br The buffer resource to use for memory allocation.
     * @param preferred_mem_type The preferred memory type to use for the concatenated
     * data buffer.
     * @return Chunk The concatenated chunk.
     * @throws std::logic_error if the input vector is empty.
     */
    static Chunk concat(
        std::vector<Chunk>&& chunks,
        ChunkID chunk_id,
        rmm::cuda_stream_view stream,
        BufferResource* br,
        std::optional<MemoryType> preferred_mem_type = std::nullopt
    );

  private:
    // constructor
    Chunk(
        ChunkID chunk_id,
        std::vector<PartID> part_ids,
        std::vector<size_t> expected_num_chunks,
        std::vector<uint32_t> meta_offsets,
        std::vector<uint64_t> data_offsets,
        std::unique_ptr<std::vector<uint8_t>> metadata = nullptr,
        std::unique_ptr<Buffer> data = nullptr
    );

    ChunkID const chunk_id_;  ///< The ID of the chunk.
    std::vector<PartID> const
        part_ids_;  ///< The partition IDs of the messages in the chunk.
    std::vector<size_t> const expected_num_chunks_;  ///< The expected number of chunks of
                                                     ///< the messages in the chunk.
    std::vector<uint32_t> const
        meta_offsets_;  ///< The offsets of the metadata of the messages in the chunk.
    std::vector<uint64_t> const
        data_offsets_;  ///< The offsets of the data of the messages in the chunk.

    /// Metadata buffer that contains information about the messages in the chunk.
    std::unique_ptr<std::vector<uint8_t>> metadata_;

    /// Concatenated data buffer of the messages in the chunk.
    std::unique_ptr<Buffer> data_;
};

/**
 * @brief Represents a message indicating readiness to receive data for a specific chunk.
 */
class ReadyForDataMessage {
  public:
    ChunkID cid;  ///< Chunk ID associated with the message.

    /**
     * @brief Serializes the message into a byte array.
     *
     * @return A serialized byte vector representing the message.
     */
    [[nodiscard]] std::unique_ptr<std::vector<uint8_t>> pack() {
        auto msg = std::make_unique<std::vector<uint8_t>>(sizeof(ChunkID));
        std::memcpy(msg->data(), &cid, sizeof(cid));
        return msg;
    }

    /**
     * @brief Deserializes a message from a byte array.
     *
     * @param msg A serialized message byte vector.
     * @return A `ReadyForDataMessage` object.
     */
    [[nodiscard]] static ReadyForDataMessage unpack(
        std::unique_ptr<std::vector<uint8_t>> const& msg
    ) {
        ChunkID cid;
        std::memcpy(&cid, msg->data(), sizeof(cid));
        return ReadyForDataMessage{cid};
    }

    /**
     * @brief Returns a description of this instance.
     * @return The description.
     */
    [[nodiscard]] std::string str() const {
        std::stringstream ss;
        ss << "ReadyForDataMessage(cid=" << cid << ")";
        return ss.str();
    }
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
inline std::ostream& operator<<(std::ostream& os, Chunk const& obj) {
    os << obj.str();
    return os;
}

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
inline std::ostream& operator<<(std::ostream& os, ReadyForDataMessage const& obj) {
    os << obj.str();
    return os;
}

}  // namespace detail
}  // namespace rapidsmpf::shuffler
