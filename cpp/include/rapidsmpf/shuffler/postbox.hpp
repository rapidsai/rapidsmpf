/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>

namespace rapidsmpf::shuffler::detail {

/**
 * @brief A thread-safe container for managing and retrieving data chunks by partition and
 * chunk ID.
 *
 * @tparam KeyType The type of the key used to map chunks.
 */
template <typename KeyType>
class PostBox {
  public:
    using key_type = KeyType;  ///< The type of the key used to map chunks.

    /**
     * @brief Construct a new PostBox.
     *
     * @tparam Fn The type of the function that maps a partition ID to a key.
     * @param key_map_fn A function that maps a partition ID to a key.
     * @param num_keys_hint The number of keys to reserve space for.
     *
     * @note The `key_map_fn` must be convertible to a function that takes a `PartID` and
     * returns a `KeyType`.
     */
    template <typename Fn>
    PostBox(Fn&& key_map_fn, size_t num_keys_hint = 0)
        : key_map_fn_(std::move(key_map_fn)) {
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
     * @brief Marks a partition as empty.
     *
     * @param pid The ID of the partition to mark as empty.
     *
     * @throws std::logic_error If the partition ID is already in the postbox and is not
     * empty.
     */
    void mark_empty(PartID pid);

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
    std::unordered_map<ChunkID, Chunk> extract_by_key(KeyType key);

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
    [[nodiscard]] std::vector<std::tuple<key_type, ChunkID, std::size_t>> search(
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
    std::function<key_type(PartID)>
        key_map_fn_;  ///< Function to map partition IDs to keys.
    std::unordered_map<key_type, std::unordered_map<ChunkID, Chunk>>
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
template <typename KeyType>
inline std::ostream& operator<<(std::ostream& os, PostBox<KeyType> const& obj) {
    os << obj.str();
    return os;
}

}  // namespace rapidsmpf::shuffler::detail
