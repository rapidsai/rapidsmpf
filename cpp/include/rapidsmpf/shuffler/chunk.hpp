/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>

#include <cuda/std/span>

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/shuffler/partition.hpp>

namespace rapidsmpf::shuffler::detail {

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
    // friend the builder class
    friend class ChunkBuilder;

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
    constexpr ChunkID chunk_id() const {
        return chunk_id_;
    }

    /**
     * @brief The number of messages in the chunk.
     *
     * @return The number of messages in the chunk.
     */
    inline size_t n_messages() const {
        return part_ids_.size();
    }

    /**
     * @brief Partition ID of the i-th message.
     *
     * @param i The index of the message.
     * @return The ID of the partition.
     */
    inline PartID part_id(size_t i) const {
        return part_ids_.at(i);
    }

    /**
     * @brief The expected number of chunks of the i-th message.
     *
     * @param i The index of the message.
     * @return The expected number of chunks for the message. Non-zero when the message
     * is a control message, otherwise zero (data message).
     */
    inline size_t expected_num_chunks(size_t i) const {
        return expected_num_chunks_.at(i);
    }

    /**
     * @brief Whether the i-th message is a control message.
     *
     * @param i The index of the message.
     * @return True if the message is a control message, false otherwise.
     */
    inline bool is_control_message(size_t i) const {
        // expected_num_chunks is always non-zero for control messages
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
     * Otherwise a new chunk will be created by copying data.
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
    inline uint32_t metadata_size(size_t i) const {
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
    inline size_t data_size(size_t i) const {
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
    inline bool is_data_buffer_set() const {
        return data_ != nullptr;
    }

    /**
     * @brief Whether the metadata buffer is set.
     *
     * @return True if the metadata buffer is set, false otherwise.
     */
    inline bool is_metadata_buffer_set() const {
        return metadata_ != nullptr && !metadata_->empty();
    }

    /**
     * @brief Get the memory type of the data buffer.
     *
     * @return The memory type of the data buffer.
     */
    MemoryType data_memory_type() const {
        RAPIDSMPF_EXPECTS(data_, "data buffer is not set");
        return data_->mem_type();
    }

    /**
     * @brief Get the size of the concatenated data.
     *
     * @return The size of the concatenated data.
     */
    inline size_t concat_data_size() const {
        return data_offsets_[n_messages() - 1];
    }

    /**
     * @brief Get the size of the concatenated metadata.
     *
     * @return The size of the concatenated metadata.
     */
    inline size_t concat_metadata_size() const {
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
     * @param max_nbytes The maximum size of the chunk data to include.
     * @param stream The CUDA stream.
     * @return The description.
     */
    [[nodiscard]] std::string str(
        size_t max_nbytes = 512, rmm::cuda_stream_view stream = cudf::get_default_stream()
    ) const;

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
 * @brief Builder class for constructing Chunk objects.
 */
class ChunkBuilder {
  public:
    /**
     * @brief Construct a new Builder object.
     *
     * @param chunk_id The ID of the chunk.
     * @param stream The CUDA stream to use to build the chunk.
     * @param br The buffer resource to use to build the chunk.
     * @param num_messages_hint Hint for the expected number of messages in the chunk.
     *                          Used to pre-reserve vectors for better performance.
     */
    explicit ChunkBuilder(
        ChunkID chunk_id,
        rmm::cuda_stream_view stream,
        BufferResource* br,
        size_t num_messages_hint = 0
    );

    /**
     * @brief Add a control message to the chunk.
     *
     * @param part_id The partition ID of the message.
     * @param expected_num_chunks The expected number of chunks for this message.
     * @return ChunkBuilder& Reference to this builder for method chaining.
     */
    ChunkBuilder& add_control_message(PartID part_id, size_t expected_num_chunks);

    /**
     * @brief Add a data message to the chunk using packed data.
     *
     * @param part_id The partition ID of the message.
     * @param packed_data The packed data containing metadata and GPU data.
     * @return ChunkBuilder& Reference to this builder for method chaining.
     */
    ChunkBuilder& add_packed_data(PartID part_id, PackedData&& packed_data);

    /**
     * @brief Build the Chunk object. This will concatenate the staged metadata and data
     * buffers into a single metadata and data buffer.
     *
     * @return Chunk The constructed Chunk object.
     * @throws std::runtime_error if no messages are added.
     */
    Chunk build();

  private:
    ChunkID chunk_id_;
    rmm::cuda_stream_view stream_;
    BufferResource* br_;
    std::vector<PartID> part_ids_;
    std::vector<size_t> expected_num_chunks_;
    std::vector<uint32_t> meta_offsets_;
    std::vector<uint64_t> data_offsets_;
    std::vector<std::vector<uint8_t>>
        staged_metadata_;  ///< Temporary storage for metadata during building

    std::vector<std::unique_ptr<Buffer>>
        staged_data_;  ///< Temporary storage for GPU data during building
};

/**
 * @brief Represents a message indicating readiness to receive data for a specific chunk.
 */
class ReadyForDataMessage {
  public:
    PartID pid;  ///< Partition ID associated with the message.
    ChunkID cid;  ///< Chunk ID associated with the message.

    /**
     * @brief Serializes the message into a byte array.
     *
     * @return A serialized byte vector representing the message.
     */
    [[nodiscard]] std::unique_ptr<std::vector<uint8_t>> pack() {
        auto msg = std::make_unique<std::vector<uint8_t>>(sizeof(ReadyForDataMessage));
        *reinterpret_cast<ReadyForDataMessage*>(msg->data()) = {pid, cid};
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
        return *reinterpret_cast<ReadyForDataMessage const*>(msg->data());
    }

    /**
     * @brief Returns a description of this instance.
     * @return The description.
     */
    [[nodiscard]] std::string str() const {
        std::stringstream ss;
        ss << "ReadyForDataMessage(pid=" << pid << ", cid=" << cid << ")";
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

}  // namespace rapidsmpf::shuffler::detail
