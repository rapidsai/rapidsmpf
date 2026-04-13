/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>

namespace rapidsmpf::shuffler::detail {

/**
 * @brief A thread-safe container for managing outgoing (to send) chunks.
 */
class ChunksToSend {
  public:
    ChunksToSend() = default;

    /**
     * @brief Insert a chunk into the container.
     *
     * @param c The chunk to insert.
     */
    void insert(std::unique_ptr<Chunk> c);

    /**
     * @brief Extract ready chunks.
     *
     * @note Ready means no stream-ordered work queued on the chunk's data.
     *
     * @return Vector of chunks ready to send.
     */
    [[nodiscard]] std::vector<Chunk> extract_ready();

    /**
     * @brief @return Whether the container is empty.
     */
    [[nodiscard]] bool empty() const;

    /**
     * @brief @return Returns a description of this instance.
     */
    [[nodiscard]] std::string str() const;

  private:
    mutable std::mutex mutex_{};
    std::vector<std::unique_ptr<Chunk>> chunks_{};
};

/**
 * @brief Overloads the stream insertion operator for the ChunksToSend class.
 *
 * This function allows a description of ChunksToSend to be written to an output stream.
 *
 * @param os The output stream to write to.
 * @param obj The object to write.
 * @return A reference to the modified output stream.
 */
inline std::ostream& operator<<(std::ostream& os, ChunksToSend const& obj) {
    os << obj.str();
    return os;
}

/**
 * @brief A thread-safe container for managing received chunks stratified by partition ID.
 */
class ReceivedChunks {
  public:
    /**
     * @brief Construct a new container.
     *
     * @param num_keys_hint The number of keys to reserve space for.
     */
    ReceivedChunks(std::size_t num_keys_hint = 0) {
        if (num_keys_hint > 0) {
            pigeonhole_.reserve(num_keys_hint);
        }
    }

    /**
     * @brief Insert a chunk.
     *
     * @param chunk The chunk to insert.
     */
    void insert(Chunk&& chunk);

    /**
     * @brief Check whether the specified partition contains any chunks.
     *
     * @param pid Identifier of the partition to query.
     * @return True if the partition contains no chunks, false otherwise.
     *
     * @note The result reflects a snapshot at the time of the call and may change
     * immediately afterward.
     */
    [[nodiscard]] bool is_empty(PartID pid) const;

    /**
     * @brief Extracts all chunks associated with a specific partition.
     *
     * @param pid The ID of the partition.
     * @return A vector of chunks.
     *
     * @throws std::out_of_range If the partition is not found.
     */
    [[nodiscard]] std::vector<Chunk> extract(PartID pid);

    /**
     * @brief Checks if the container is empty.
     *
     * @return `true` if the container is empty, `false` otherwise.
     *
     * @note The result reflects a snapshot at the time of the call and may change
     * immediately afterward.
     */
    [[nodiscard]] bool empty() const;

    /**
     * @brief @return A description of this container.
     */
    [[nodiscard]] std::string str() const;

    /**
     * @brief Spill device data.
     *
     * The spilling is stream ordered by the spilled buffers' CUDA streams.
     *
     * @param br The buffer resource for host and device allocations.
     * @param amount Requested amount of data to spill in bytes.
     * @return Actual amount of data spilled in bytes.
     */
    [[nodiscard]] std::size_t spill(BufferResource* br, std::size_t amount);

  private:
    // TODO: more fine-grained locking e.g. by locking each partition individually.
    mutable std::mutex mutex_;
    std::unordered_map<PartID, std::vector<Chunk>>
        pigeonhole_;  ///< Storage for chunks, stratified by partition ID.
};

/**
 * @brief Overloads the stream insertion operator for the ReceivedChunks class.
 *
 * This function allows a description of ReceivedChunks be written to an output stream.
 *
 * @param os The output stream to write to.
 * @param obj The object to write.
 * @return A reference to the modified output stream.
 */
inline std::ostream& operator<<(std::ostream& os, ReceivedChunks const& obj) {
    os << obj.str();
    return os;
}

}  // namespace rapidsmpf::shuffler::detail
