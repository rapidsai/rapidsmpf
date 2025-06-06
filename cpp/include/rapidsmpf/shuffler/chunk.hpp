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
     * @param gpu_data_size If known, the size of the gpu data buffer (in bytes).
     * @param metadata The metadata of the packed `cudf::table` that makes up this
     * chunk.
     *  @param gpu_data The gpu_data of the packed `cudf::table` that makes up this
     * chunk.
     */
    Chunk(
        PartID pid,
        ChunkID cid,
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

    /**
     * @brief Returns true if the chunk is ready for consumption.
     *
     * Checks that the gpu_data's CUDA event is ready, if gpu_data contains a valid
     * buffer. The CUDA event is used to synchronize the chunk's data to ensure
     * any allocation or copy (e.g., spilling) is complete before the chunk is
     * consumed. If expected_num_chunks is greater than 0, or gpu_data_size is 0,
     * the chunk is considered always ready as it should not have any CUDA data
     * to receive.
     *
     * @return true if the chunk is ready, false otherwise.
     */
    [[nodiscard]] bool is_ready() const;

  private:
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
