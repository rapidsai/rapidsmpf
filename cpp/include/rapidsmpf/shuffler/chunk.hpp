/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>
#include <sstream>
#include <vector>

#include <cuda/std/span>

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/shuffler/partition.hpp>

namespace rapidsmpf::shuffler::detail {

/**
 * @brief The globally unique ID of a chunk.
 */
using ChunkID = std::uint64_t;

/**
 * Format:
 * - chunk_id: uint64_t, ID of the chunk
 * - n_elements: size_t, Number of messages in the chunk
 * - [partition_ids]: std::vector<PartID>, Partition IDs of the messages, size =
 * n_elements
 * - [expected_num_chunks]: std::vector<size_t>, Expected number of chunks of the
 * messages, size = n_elements
 * - [psum_meta]: std::vector<uint32_t>, Prefix sums (excluding 0) of the metadata
 * sizes of the messages, size = n_elements
 * - [psum_data]: std::vector<uint64_t>, Prefix sums (excluding 0) of the data sizes of
 * the messages, size = n_elements
 * - [concat_metadata]: std::vector<uint8_t>, Concatenated metadata of the messages,
 * size = psum_meta[n_elements - 1]
 *
 * For a chunk with N messages with M bytes of concat metadata the size of metadata_
 * buffer is sizeof(ChunkID) + sizeof(size_t) + N * (sizeof(PartID) + sizeof(size_t) +
 * sizeof(uint32_t) + sizeof(uint64_t)) + M = 16 + N * 24 + M bytes.
 *
 * For a chunk with a single control message, the size of the metadata_ buffer is
 * sizeof(ChunkID) + sizeof(PartID)+ 2*sizeof(size_t) + sizeof(uint32_t) +
 * sizeof(uint64_t) = 40 bytes.
 *
 * For a chunk with a single message with M bytes of metadata, the size of the metadata_
 * buffer is sizeof(ChunkID) + sizeof(PartID) + sizeof(size_t) + sizeof(uint32_t) +
 * sizeof(ChunkID) + sizeof(PartID) + sizeof(size_t) + sizeof(uint32_t) + sizeof(uint64_t)
 * + M = 40 + M bytes.
 */
class ChunkBatch {
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
    inline ChunkID chunk_id() const {
        return *reinterpret_cast<ChunkID*>(metadata_->data());
    }

    /**
     * @brief The number of messages in the chunk.
     *
     * @return The number of messages in the chunk.
     */
    inline size_t n_messages() const {
        return *reinterpret_cast<size_t*>(metadata_->data() + sizeof(ChunkID));
    }

    /**
     * @brief Partition ID of the i-th message.
     *
     * @param i The index of the message.
     * @return The ID of the partition.
     */
    inline PartID part_id(size_t i) const {
        return *(part_ids_begin() + i);
    }

    /**
     * @brief The expected number of chunks of the i-th message.
     *
     * @param i The index of the message.
     * @return The expected number of chunks for the message. Non-zero when the message
     * is a control message, otherwise zero (data message).
     */
    inline size_t expected_num_chunks(size_t i) const {
        return *(expected_num_chunks_begin() + i);
    }

    /**
     * @brief Whether the i-th message is a control message.
     *
     * @param i The index of the message.
     * @return True if the message is a control message, false otherwise.
     */
    inline bool is_control_message(size_t i) const {
        return expected_num_chunks(i) > 0;
    }

    /**
     * @brief Get the data of the i-th message, as a new ChunkBatch.
     *
     * @param i The index of the message.
     * @return A new ChunkBatch containing the data of the i-th message.
     * @note This will create a copy of the packed data. If i==0 and n_messages() == 1 and
     * the message is a data message, the data buffer will be moved to the new ChunkBatch.
     */
    ChunkBatch get_data(ChunkID new_chunk_id, size_t i, rmm::cuda_stream_view stream);

    /**
     * @brief Get the size of the metadata of the i-th message.
     *
     * @param i The index of the message.
     * @return The size of the metadata of the message. Zero when the message is a
     * control message, otherwise the size of `PackedData::metadata`.
     */
    inline uint32_t metadata_size(size_t i) const {
        return i == 0 ? *(psum_meta_begin())
                      : *(psum_meta_begin() + i) - *(psum_meta_begin() + i - 1);
    }

    /**
     * @brief Get the size of the packed data of the i-th message.
     *
     * @param i The index of the message.
     * @return The size of the packed data of the message. Zero when the message is a
     * control message, otherwise the size of `PackedData::gpu_data` of the message.
     */
    inline size_t data_size(size_t i) const {
        return i == 0 ? *(psum_data_begin())
                      : *(psum_data_begin() + i) - *(psum_data_begin() + i - 1);
    }

    /**
     * @brief Set the data buffer.
     *
     * @param data The data buffer.
     */
    void set_data_buffer(std::unique_ptr<Buffer> data) {
        data_ = std::move(data);
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
     * @brief Create a single-message ChunkBatch from a packed data.
     *
     * @param chunk_id The ID of the chunk.
     * @param part_id The ID of the partition.
     * @param packed_data The packed data.
     * @param br The buffer resource.
     * @return The ChunkBatch.
     */
    static ChunkBatch from_packed_data(
        ChunkID chunk_id, PartID part_id, PackedData&& packed_data, BufferResource* br
    );

    /**
     * @brief Create a single-message ChunkBatch for a finished partition (control
     * message).
     *
     * @param chunk_id The ID of the chunk.
     * @param part_id The ID of the partition.
     * @param expected_num_chunks The expected number of chunks.
     * @return The ChunkBatch.
     */
    static ChunkBatch from_finished_partition(
        ChunkID chunk_id, PartID part_id, size_t expected_num_chunks
    );

    /**
     * @brief Create a ChunkBatch from a metadata message.
     *
     * @param msg The metadata message received from another rank.
     * @param validate Whether to validate the metadata buffer.
     * @return The ChunkBatch.
     *
     * @throws std::runtime_error if the metadata buffer does not follow the expected
     * format and `validate` is true.
     */
    static ChunkBatch from_metadata_message(
        std::unique_ptr<std::vector<uint8_t>> msg, bool validate = true
    );


    /**
     * @brief Validate if a provided metadata buffer follows the expected format.
     *
     * @param metadata_buf The metadata buffer to validate.
     * @return True if the metadata buffer follows the expected format, false otherwise.
     */
    static bool validate_metadata_format(std::vector<uint8_t> const& metadata_buf);


  private:
    /// @brief The beginning of the partition IDs in the chunk.
    inline PartID* part_ids_begin() const {
        return reinterpret_cast<PartID*>(
            metadata_->data() + sizeof(ChunkID) + sizeof(size_t)
        );
    }

    /// @brief The beginning of the expected number of chunks in the chunk.
    inline size_t* expected_num_chunks_begin() const {
        return reinterpret_cast<size_t*>(part_ids_begin() + n_messages());
    }

    /// @brief The beginning of the psum metadata in the chunk.
    inline uint32_t* psum_meta_begin() const {
        return reinterpret_cast<uint32_t*>(expected_num_chunks_begin() + n_messages());
    }

    /// @brief The beginning of the psum data in the chunk.
    inline uint64_t* psum_data_begin() const {
        return reinterpret_cast<uint64_t*>(psum_meta_begin() + n_messages());
    }

    /// @brief The beginning of the concat metadata in the chunk.
    inline uint8_t* concat_metadata_begin() const {
        return reinterpret_cast<uint8_t*>(psum_data_begin() + n_messages());
    }

    /// @brief The size of the concat metadata in the chunk.
    inline size_t concat_metadata_size() const {
        return *(psum_data_begin() + n_messages() - 1);
    }

    /// Metadata buffer that contains information about the messages in the chunk.
    std::unique_ptr<std::vector<uint8_t>> metadata_;

    /// Concatenated data buffer of the messages in the chunk.
    std::unique_ptr<Buffer> data_;
};

/**
 * @brief A chunk of a partition.
 */
class Chunk {
  public:
    PartID const pid;  ///< Partition ID that this chunk belongs to.
    ChunkID const cid;  ///< Unique ID of this chunk.

    /// If not zero, the number of chunks of the partition expected to get from the
    /// sending rank. Ignored when it is zero.
    std::size_t const expected_num_chunks;

    /// If known, the size of the GPU data buffer (in bytes).
    std::size_t const gpu_data_size;

    /// Metadata of the packed `cudf::table` associated with this chunk.
    std::unique_ptr<std::vector<uint8_t>> metadata;

    /// GPU data buffer of the packed `cudf::table` associated with this chunk.
    std::unique_ptr<Buffer> gpu_data;

    /**
     * @brief Construct a new chunk of a partition.
     *
     * @param pid The ID of the partition this chunk is part of.
     * @param cid The ID of the chunk.
     * @param expected_num_chunks If not zero, the number of chunks of the partition
     * expected to get from the sending rank. Ignored when it is zero.
     * @param gpu_data_size If known, the size of the gpu data buffer (in bytes).
     * @param metadata The metadata of the packed `cudf::table` that makes up this
     * chunk.
     *  @param gpu_data The gpu_data of the packed `cudf::table` that makes up this
     * chunk.
     */
    Chunk(
        PartID pid,
        ChunkID cid,
        std::size_t expected_num_chunks,
        std::size_t gpu_data_size,
        std::unique_ptr<std::vector<uint8_t>> metadata,
        std::unique_ptr<Buffer> gpu_data
    );

    /**
     * @brief Construct a new chunk of a partition.
     *
     * @param pid The ID of the partition this chunk is part of.
     * @param cid The ID of the chunk.
     * @param expected_num_chunks If not zero, the number of chunks of the partition
     * expected to get from the sending rank. Ignored when it is zero.
     */
    Chunk(PartID pid, ChunkID cid, std::size_t expected_num_chunks);

    /**
     * @brief Header of a metadata message.
     */
    struct MetadataMessageHeader {
        PartID pid;  ///< The ID of the partition this chunk is part of.
        ChunkID cid;  ///< The ID of the chunk.
        /// If not zero, the number of chunks of the partition expected to get from the
        /// sending rank. Ignored when it is zero.
        std::size_t expected_num_chunks;
        /// If known, the size of the gpu data buffer (in bytes).
        std::size_t gpu_data_size;
    };

    /**
     * @brief Returns a metadata message that represents this chunk.
     *
     * @returns The metadata message as a serialized byte vector.
     */
    [[nodiscard]] std::unique_ptr<std::vector<uint8_t>> to_metadata_message() const;

    /**
     * @brief Construct a new chunk from a metadata message.
     *
     * @param msg A serialized metadata message previously returned by
     * `to_metadata_message`.
     * @returns The new chunk.
     */
    [[nodiscard]] static Chunk from_metadata_message(
        std::unique_ptr<std::vector<uint8_t>> const& msg
    );

    /**
     * @brief Returns an unpacked (deserialized) chunk.
     *
     * @warning This copies the data and shouldn't be used in performance critical code.
     *
     * @param stream CUDA stream used for device memory operations and kernel launches.
     * @returns A `cudf::table` that represents the chunk data.
     */
    [[nodiscard]] std::unique_ptr<cudf::table> unpack(rmm::cuda_stream_view stream) const;

    /**
     * @brief Returns a description of this instance.
     *
     * @param max_nbytes The maximum size of the chunk data to include.
     * @param stream CUDA stream used for device memory operations and kernel launches.
     * @return The description.
     */
    [[nodiscard]] std::string str(
        std::size_t max_nbytes = 512,
        rmm::cuda_stream_view stream = cudf::get_default_stream()
    ) const;
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
