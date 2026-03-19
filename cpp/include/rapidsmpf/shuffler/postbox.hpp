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
 * @brief A thread-safe container for managing outgoing chunks.
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
 * @brief A thread-safe container for managing and retrieving data chunks by partition and
 * chunk ID.
 */
class PostBox {
  public:
    /**
     * @brief Construct a new PostBox.
     *
     * @param num_keys_hint The number of keys to reserve space for.
     */
    PostBox(std::size_t num_keys_hint = 0) {
        if (num_keys_hint > 0) {
            pigeonhole_.reserve(num_keys_hint);
        }
    }

    /**
     * @brief Inserts a chunk into the PostBox.
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
    bool is_empty(PartID pid) const;

    /**
     * @brief Extracts a specific chunk from the PostBox.
     *
     * @param pid The ID of the partition containing the chunk.
     * @param cid The ID of the chunk to be accessed.
     * @return The extracted chunk.
     *
     * @throws std::out_of_range If the chunk is not found.
     */
    [[nodiscard]] Chunk extract(PartID pid, ChunkID cid);

    /**
     * @brief Extracts all chunks associated with a specific partition.
     *
     * @param pid The ID of the partition.
     * @return A map of chunk IDs to chunks for the specified partition.
     *
     * @throws std::out_of_range If the partition is not found.
     */
    std::unordered_map<ChunkID, Chunk> extract(PartID pid);

    /**
     * @brief Extracts all chunks associated with a specific key.
     *
     * @param key The key.
     * @return A map of chunk IDs to chunks for the specified key.
     *
     * @throws std::out_of_range If the key is not found.
     */
    std::unordered_map<ChunkID, Chunk> extract_by_key(PartID key);

    /**
     * @brief Extracts all ready chunks from the PostBox.
     *
     * @return A vector of all ready chunks in the PostBox.
     */
    std::vector<Chunk> extract_all_ready();

    /**
     * @brief Checks if the PostBox is empty.
     *
     * @return `true` if the PostBox is empty, `false` otherwise.
     */
    [[nodiscard]] bool empty() const;

    /**
     * @brief Searches for chunks of the specified memory type.
     *
     * @param mem_type The type of memory to search within.
     * @return A vector of tuples, where each tuple contains: PartID, ChunkID, and the
     * size of the chunk.
     */
    [[nodiscard]] std::vector<std::tuple<PartID, ChunkID, std::size_t>> search(
        MemoryType mem_type
    ) const;

    /**
     * @brief Returns a description of this instance.
     * @return The description.
     */
    [[nodiscard]] std::string str() const;

  private:
    // TODO: more fine-grained locking e.g. by locking each partition individually.
    mutable std::mutex mutex_;
    std::unordered_map<PartID, std::unordered_map<ChunkID, Chunk>>
        pigeonhole_;  ///< Storage for chunks, organized by a key and chunk ID.
};

/**
 * @brief Overloads the stream insertion operator for the PostBox class.
 *
 * This function allows a description of a PostBox to be written to an output stream.
 *
 * @param os The output stream to write to.
 * @param obj The object to write.
 * @return A reference to the modified output stream.
 */
inline std::ostream& operator<<(std::ostream& os, PostBox const& obj) {
    os << obj.str();
    return os;
}

}  // namespace rapidsmpf::shuffler::detail
